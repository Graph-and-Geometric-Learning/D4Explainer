import argparse
import os
import os.path as osp
import time
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear as Lin, ModuleList, ReLU, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import BatchNorm, GCNConv
from torch_geometric.utils import accuracy

from utils import set_seed

EPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tree Grids Model")

    parser.add_argument("--data_name", nargs="?", default="Tree_Grids", help="Input data path.")
    parser.add_argument(
        "--model_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "param", "gnns"),
        help="path for saving trained model.",
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--epoch", type=int, default=10000, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=128, help="hiden size.")
    parser.add_argument("--verbose", type=int, default=10, help="Interval of evaluation.")
    parser.add_argument("--num_unit", type=int, default=4, help="number of Convolution layers(units)")
    parser.add_argument(
        "--random_label", type=bool, default=False, help="train a model under label randomization for sanity check"
    )
    return parser.parse_args()


class Syn_GCN_TG(torch.nn.Module):
    def __init__(self, num_unit, n_input, n_out, n_hid):
        super(Syn_GCN_TG, self).__init__()
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()
        self.edge_emb = Lin(3, 1)
        self.relus.extend([ReLU()] * num_unit)
        self.convs.append(GCNConv(in_channels=n_input, out_channels=n_hid))
        for i in range(num_unit - 2):
            self.convs.append(GCNConv(in_channels=n_hid, out_channels=n_hid))
        self.convs.append(GCNConv(in_channels=n_hid, out_channels=n_hid))
        self.batch_norms.extend([BatchNorm(n_hid)] * num_unit)

        # self.lin1 = Lin(128, 128)
        self.ffn = nn.Sequential(*([nn.Linear(n_hid, n_hid)] + [ReLU(), nn.Dropout(), nn.Linear(n_hid, n_out)]))

        self.softmax = Softmax(dim=1)
        self.dropout = 0.6

    def forward(self, x, edge_index, edge_attr=None):
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index)
            x = relu(x)
        pred = self.ffn(x)  # [node_num, n_class]
        self.readout = self.softmax(pred)
        return pred

    def get_node_reps(self, x, edge_index, edge_attr=None):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight)
            x = relu(x)
        node_x = x
        return node_x

    def get_node_pred_subgraph(self, x, edge_index, mapping=None):
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index)
            # x = F.dropout(x, self.dropout, training=self.training)
            # x = relu(batch_norm(x))
            x = relu(x)
        node_repr = self.ffn(x)  # [node_num, n_class]
        node_prob = self.softmax(node_repr)
        output_prob = node_prob[mapping]  # [bsz, n_classes]
        output_repr = node_repr[mapping]  # [bsz, n_classes]
        return output_prob, output_repr

    def get_pred_explain(self, x, edge_index, edge_mask, mapping=None):
        edge_mask = (edge_mask * EPS).sigmoid()
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight=edge_mask)
            # x = F.dropout(x, self.dropout, training=self.training)
            # x = relu(batch_norm(x))
            x = relu(x)
        node_repr = self.ffn(x)  # [node_num, n_class]
        node_prob = self.softmax(node_repr)
        output_prob = node_prob[mapping]  # [bsz, n_classes]
        output_repr = node_repr[mapping]  # [bsz, n_classes]
        return output_prob, output_repr

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)



if __name__ == "__main__":
    set_seed(33)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    name = args.data_name
    file_dir = osp.join(osp.dirname(__file__), "..", "data", name, "processed/whole_graph.pt")
    data = torch.load(file_dir)
    data.to(device)
    n_input = data.x.size(1)
    n_labels = int(torch.unique(data.y).size(0))
    model = Syn_GCN_TG(args.num_unit, n_input=n_input, n_out=n_labels, n_hid=args.hidden).to(device)

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
    print("y_pred_orig counts: {}".format(np.unique(pred, return_counts=True)))
