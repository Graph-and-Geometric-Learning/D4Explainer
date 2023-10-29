import torch
import torch.nn as nn
import torch.nn.functional as F

SLOPE = 0.01


def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    Instance normalization for 2D feature maps with mask
    :param x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    :param mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    return: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    zero_indices = torch.where(torch.sum(mask, dim=[1, 2]) < 0.5)[0].squeeze(-1)  # [N,]
    mean = torch.sum(x * mask, dim=[1, 2]) / (torch.sum(mask, dim=[1, 2]))  # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask) ** 2  # (N,L,L,C)
    var = torch.sum(var_term, dim=[1, 2]) / (torch.sum(mask, dim=[1, 2])) + 1e-5  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)  # (N, L, L, C)
    instance_norm = instance_norm * mask
    instance_norm[zero_indices, :, :, :] = 0
    return instance_norm


class PowerfulLayer(nn.Module):
    # in_feature will be given as hidden-dim and out as well hidden-dim
    def __init__(self, in_feat: int, out_feat: int, num_layers: int, spectral_norm=(lambda x: x)):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        activation = nn.LeakyReLU(negative_slope=SLOPE)
        # generate num_layers linear layers with the first one being of dim in_feat and all others out_feat, in between those layers we add an activation layer
        self.m1 = nn.Sequential(
            *[
                spectral_norm(nn.Linear(in_feat if i == 0 else out_feat, out_feat)) if i % 2 == 0 else activation
                for i in range(num_layers * 2 - 1)
            ]
        )
        self.m2 = nn.Sequential(
            *[
                spectral_norm(nn.Linear(in_feat if i == 0 else out_feat, out_feat)) if i % 2 == 0 else activation
                for i in range(num_layers * 2 - 1)
            ]
        )
        # a linear layer that has input dim in_feat + out_feat and outputdim out_feat and a possible bias
        # this is mentioned in paper as the layer after each concatenbation back to outputdim
        self.m4 = nn.Sequential(spectral_norm(nn.Linear(in_feat + out_feat, out_feat, bias=True)))

    # expects x as batch x N x N x in_feat and mask as batch x N x N x 1
    def forward(self, x, mask):
        """x: batch x N x N x in_feat"""

        # norm by taking uppermost col of mask gives nr of nodes active and then sqrt of that and put it in matching array dim
        # [bsz, N, N, 1]
        norm = mask.squeeze(-1).float().sum((-1, -2)).sqrt().sqrt().view(mask.size(0), 1, 1, 1)

        # here I will start to treat mask as in the edp paper, namely batch x N with then a boolean value
        # batch * N * N * 1 gets to batch * 1 * N * N
        mask = mask.unsqueeze(1).squeeze(-1)

        # run the two mlp on input and permute dimensions so that now matches dim of mask: batch * N * N * out_features
        out1 = self.m1(x).permute(0, 3, 1, 2) * mask  # batch, out_feat, N, N
        out2 = self.m2(x).permute(0, 3, 1, 2) * mask  # batch, out_feat, N, N

        # matrix multiply each matching layer of features as well as adjacencies
        out = out1 @ out2
        del out1, out2
        out = out / (norm + 1e-5)
        # permute back to correct dim and concat with the skip-mlp in last dim
        out_cat = torch.cat((out.permute(0, 2, 3, 1), x), dim=3)  # batch, N, N, out_feat
        del x
        # run through last layer to go back to out_features dim
        out = self.m4(out_cat)

        return out


# this is the class for the invariant layer that makes the whole thing invariant
class FeatureExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, spectral_norm=(lambda x: x)):
        super().__init__()
        self.lin1 = nn.Sequential(spectral_norm(nn.Linear(in_features, out_features, bias=True)))
        self.lin2 = nn.Sequential(spectral_norm(nn.Linear(in_features, out_features, bias=False)))
        self.lin3 = nn.Sequential(spectral_norm(nn.Linear(out_features, out_features, bias=False)))
        self.activation = nn.LeakyReLU(negative_slope=SLOPE)

    def forward(self, u, mask):
        """
        Forward pass of the invariant layer.
        :param u: (batch_size, num_nodes, num_nodes, in_features)
        :param mask: (batch_size, num_nodes, num_nodes, 1)
        :return: (batch_size, out_features).
        """
        u = u * mask
        # tensor of batch * 1 that represernts nr of active nodes
        n = mask[:, 0].sum(1) + 1e-5
        # tensor of batches * features * their diagonal elements (this retrieves the node elements that are stored on the diagonal)
        diag = u.diagonal(dim1=1, dim2=2)  # batch_size, channels, num_nodes

        # tensor of batch * features with storing the sum of diagonals
        trace = torch.sum(diag, dim=2)
        del diag

        out1 = self.lin1.forward(trace / n)
        s = (torch.sum(u, dim=[1, 2]) - trace) / (n * (n - 1))
        del trace

        out2 = self.lin2.forward(s)  # bs, out_feat
        del s

        out = out1 + out2
        out = out + self.lin3.forward(self.activation(out))
        return out


class Powerful(nn.Module):
    def __init__(
        self,
        args,
        spectral_norm=(lambda x: x),
        project_first: bool = False,
        node_out: bool = False,
    ):
        super().__init__()
        self.cat_output = args.cat_output
        self.normalization = args.normalization
        self.layers_per_conv = args.layers_per_conv  # was 1 originally, try 2?
        self.layer_after_conv = args.simplified
        self.dropout_p = args.dropout
        self.residual = args.residual
        # self.activation = nn.LeakyReLU(negative_slope=SLOPE)
        self.activation = nn.ReLU()
        self.project_first = project_first
        self.node_out = node_out
        self.output_features = 1
        self.node_output_features = 1
        self.noise_mlp = args.noise_mlp
        self.device = args.device
        self.num_layers = args.num_layers
        self.hidden = args.n_hidden

        self.time_mlp = nn.Sequential(nn.Linear(1, 4), nn.GELU(), nn.Linear(4, 1))
        self.input_features = 2 * args.feature_in + 2

        self.in_lin = nn.Sequential(spectral_norm(nn.Linear(self.input_features, self.hidden)))

        if self.cat_output:
            if self.project_first:
                self.layer_cat_lin = nn.Sequential(
                    spectral_norm(nn.Linear(self.hidden * (self.num_layers + 1), self.hidden))
                )
            else:
                self.layer_cat_lin = nn.Sequential(
                    spectral_norm(nn.Linear(self.hidden * self.num_layers + self.input_features, self.hidden))
                )

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        for _ in range(self.num_layers):
            self.convs.append(
                PowerfulLayer(self.hidden, self.hidden, self.layers_per_conv, spectral_norm=spectral_norm)
            )

        self.feature_extractors = torch.nn.ModuleList([])
        for _ in range(self.num_layers):
            if self.normalization == "batch":
                self.bns.append(nn.BatchNorm2d(self.hidden))
            else:
                self.bns.append(None)
            self.feature_extractors.append(FeatureExtractor(self.hidden, self.hidden, spectral_norm=spectral_norm))
        if self.layer_after_conv:
            self.after_conv = nn.Sequential(spectral_norm(nn.Linear(self.hidden, self.hidden)))
        self.final_lin = nn.Sequential(spectral_norm(nn.Linear(self.hidden, self.output_features)))

        if self.node_out:
            if self.cat_output:
                if self.project_first:
                    self.layer_cat_lin_node = nn.Sequential(
                        spectral_norm(nn.Linear(self.hidden * (self.num_layers + 1), self.hidden))
                    )
                else:
                    self.layer_cat_lin_node = nn.Sequential(
                        spectral_norm(nn.Linear(self.hidden * self.num_layers + self.input_features, self.hidden))
                    )

            if self.layer_after_conv:
                self.after_conv_node = nn.Sequential(spectral_norm(nn.Linear(self.hidden, self.hidden)))
            self.final_lin_node = nn.Sequential(spectral_norm(nn.Linear(self.hidden, self.node_output_features)))

        self.test_lin = nn.Sequential(spectral_norm(nn.Linear(self.input_features, self.output_features, bias=False)))

    def get_out_dim(self):
        """
        returns the output dimension of the model
        :return: number of output features
        """
        return self.output_features

    # expects the input as the adjacency tensor: batchsize x N x N
    # expects the node_features as tensor: batchsize x N x node_features
    # expects the mask as tensor: batchsize x N x N
    # expects noiselevel as the noislevel that was used as single float
    def forward(self, node_features, A, mask, noiselevel):
        """
        forward pass of the model
        :param node_features: [batchsize, N, C]
        :param A: [batchsize, N, N]
        :param mask: [batchsize, N, N]
        :param noiselevel: single float
        :return: [batchsize, N, N, 1]
        """
        if len(mask.shape) < 4:
            mask = mask[..., None]
        else:
            mask = mask
        if len(A.shape) < 4:
            u = A[..., None]  # [batch, N, N, 1]
        else:
            u = A

        if self.noise_mlp:
            noiselevel = torch.tensor([float(noiselevel)]).to(self.device)
            noiselevel = self.time_mlp(noiselevel)
            noise_level_matrix = noiselevel.expand(u.size(0), u.size(1), u.size(3)).to(self.device)
            noise_level_matrix = torch.diag_embed(noise_level_matrix.transpose(-2, -1), dim1=1, dim2=2)
        else:
            noiselevel = torch.full([1], noiselevel).to(self.device)
            noise_level_matrix = noiselevel.expand(u.size(0), u.size(1), u.size(3)).to(self.device)  # [bsz, N, 1]
            noise_level_matrix = torch.diag_embed(noise_level_matrix.transpose(-2, -1), dim1=1, dim2=2)

        node_feature1 = node_features.unsqueeze(1).repeat(1, node_features.size(1), 1, 1)
        node_feature2 = node_features.unsqueeze(2).repeat(1, 1, node_features.size(1), 1)
        u = torch.cat([u, node_feature1, node_feature2, noise_level_matrix], dim=-1).to(self.device)
        del node_features

        if self.project_first:
            u = self.in_lin(u)
            out = [u]
        else:
            out = [u]
            u1 = self.in_lin(u)
        for conv, bn in zip(self.convs, self.bns):
            u1 = conv(u1, mask) + (u1 if self.residual else 0)
            if self.normalization == "none":
                u1 = u1
            elif self.normalization == "instance":
                u1 = masked_instance_norm2D(u1, mask)
            elif self.normalization == "batch":
                u1 = bn(u1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            else:
                raise ValueError

            u1 = self.activation(u1)
            u2 = u1 * mask
            out.append(u2)

        out = torch.cat(out, dim=-1)
        if self.node_out:
            node_out = self.layer_cat_lin_node(out.diagonal(dim1=1, dim2=2).transpose(-2, -1))
            if self.layer_after_conv:
                node_out = node_out + self.activation(self.after_conv_node(node_out))
            node_out = F.dropout(node_out, p=self.dropout_p, training=self.training)
            node_out = self.final_lin_node(node_out)
        out = self.layer_cat_lin(out)
        out = masked_instance_norm2D(self.activation(out), mask)

        if self.layer_after_conv:
            out = out + self.activation(self.after_conv(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = self.final_lin(out)
        out = out * mask
        if self.node_out:
            return out, node_out
        else:
            return out
