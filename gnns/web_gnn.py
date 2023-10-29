import argparse
import math
import os
import os.path as osp
import time
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import BatchNorm, MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import accuracy, add_remaining_self_loops
from torch_scatter import scatter_add

from utils import set_seed

EPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train Cornell Model")

    parser.add_argument("--data_name", nargs="?", default="cornell", help="Input data path.")
    parser.add_argument(
        "--model_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "param", "gnns"),
        help="path for saving trained model.",
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--epoch", type=int, default=3000, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=32, help="hidden size.")
    parser.add_argument("--verbose", type=int, default=10, help="Interval of evaluation.")
    parser.add_argument("--num_unit", type=int, default=6, help="number of Convolution layers(units)")
    parser.add_argument(
        "--random_label", type=bool, default=False, help="train a model under label randomization for sanity check"
    )
    return parser.parse_args()


class SReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc, bias):
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(torch.Tensor(nc))
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, bias)

    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias


class EGNNConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        c_max=1.0,
        improved=False,
        cached=False,
        bias=True,
        **kwargs,
    ):
        super(EGNNConv, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = False
        self.weight = Parameter(torch.eye(in_channels) * math.sqrt(c_max))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, x_0=None, beta=0.0, residual_weight=0.0, edge_weight=None):
        """"""
        x_input = x
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}. Please "
                    "disable the caching behavior of this layer by removing "
                    "the `cached=True` argument in its constructor.".format(self.cached_num_edges, edge_index.size(1))
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        x = self.propagate(edge_index, x=x, norm=norm)
        x = (1 - residual_weight - beta) * x + residual_weight * x_input + beta * x_0
        x = torch.matmul(x, self.weight)
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class EGNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, num_layers, dropout=0.6):
        super(EGNN, self).__init__()
        # self.dataset = args.dataset
        self.num_layers = num_layers
        self.num_feats = num_node_features
        self.num_classes = num_classes
        self.dim_hidden = hidden_channels

        self.cached = False
        self.layers_GCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.layers_bn.extend([BatchNorm(self.dim_hidden)] * self.num_layers)
        self.c_min = 0.2
        self.c_max = 1
        self.beta = 0.1

        self.bias_SReLU = -10
        self.dropout = dropout
        self.output_dropout = 0.6

        self.reg_params = []
        for i in range(self.num_layers):
            c_max = self.c_max if i == 0 else 1.0
            self.layers_GCN.append(
                EGNNConv(self.dim_hidden, self.dim_hidden, c_max=c_max, cached=self.cached, bias=False)
            )
            self.layers_activation.append(SReLU(self.dim_hidden, self.bias_SReLU))
            self.reg_params.append(self.layers_GCN[-1].weight)

        self.input_layer = torch.nn.Linear(self.num_feats, self.dim_hidden)
        self.output_layer = torch.nn.Linear(self.dim_hidden, self.num_classes)
        self.non_reg_params = list(self.input_layer.parameters()) + list(self.output_layer.parameters())
        self.srelu_params = list(self.layers_activation[:-1].parameters())

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_layer(x)
        x = F.relu(x)

        original_x = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.beta

            x = self.layers_GCN[i](x, edge_index, original_x, beta=self.beta, residual_weight=residual_weight)
            # x = self.layers_bn[i](x)
            x = self.layers_activation[i](x)

        x = F.dropout(x, p=self.output_dropout, training=self.training)
        x = self.output_layer(x)
        return x

    def get_node_pred_subgraph(self, x, edge_index, mapping=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_layer(x)
        x = F.relu(x)

        original_x = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.beta
            x = self.layers_GCN[i](x, edge_index, original_x, beta=self.beta, residual_weight=residual_weight)
            x = self.layers_bn[i](x)
            x = self.layers_activation[i](x)

        x = F.dropout(x, p=self.output_dropout, training=self.training)
        node_repr = self.output_layer(x)
        node_prob = F.softmax(node_repr, dim=-1)
        output_prob = node_prob[mapping]  # [bsz, n_classes]
        output_repr = node_repr[mapping]  # [bsz, n_classes]
        return output_prob, output_repr

    def get_pred_explain(self, x, edge_index, edge_mask, mapping=None):
        edge_mask = (edge_mask * EPS).sigmoid()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_layer(x)
        x = F.relu(x)

        original_x = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.beta
            x = self.layers_GCN[i](
                x, edge_index, original_x, beta=self.beta, residual_weight=residual_weight, edge_weight=edge_mask
            )
            x = self.layers_activation[i](x)
        x = F.dropout(x, p=self.output_dropout, training=self.training)
        node_repr = self.output_layer(x)
        node_prob = F.softmax(node_repr, dim=-1)
        output_prob = node_prob[mapping]  # [bsz, n_classes]
        output_repr = node_repr[mapping]  # [bsz, n_classes]
        return output_prob, output_repr


if __name__ == "__main__":
    set_seed(44)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    name = args.data_name
    file_dir = osp.join(osp.dirname(__file__), "..", "data", name, "processed/whole_graph.pt")
    data = torch.load(file_dir)
    data.to(device)
    n_input = data.x.size(1)
    n_labels = int(torch.unique(data.y).size(0))
    model = EGNN(n_input, hidden_channels=args.hidden, num_classes=n_labels, num_layers=args.num_unit).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-5)
    min_error = None
    criterion = nn.CrossEntropyLoss()

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(x=data.x, edge_index=data.edge_index)
        loss_train = criterion(output[data.train_mask], data.y[data.train_mask])
        y_pred = torch.argmax(output, dim=1)
        acc_train = accuracy(y_pred[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimizer.step()
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(loss_train.item()),
            "acc_train: {:.4f}".format(acc_train),
            "time: {:.4f}s".format(time.time() - t),
        )

    def eval():
        model.eval()
        output = model(x=data.x, edge_index=data.edge_index)
        loss_test = criterion(output[data.test_mask], data.y[data.test_mask])
        y_pred = torch.argmax(output, dim=1)
        acc_test = accuracy(y_pred[data.test_mask], data.y[data.test_mask])
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test))
        return loss_test, y_pred

    for epoch in range(1, args.epoch + 1):
        train(epoch)

        if epoch % args.verbose == 0:
            loss_test, y_pred = eval()
            scheduler.step(loss_test)

    save_path = f"{name}_gcn.pt"

    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))
    labels = data.y[data.test_mask].cpu().numpy()
    pred = y_pred[data.test_mask].cpu().numpy()
    print("y_true counts: {}".format(np.unique(labels, return_counts=True)))
