import argparse
import os
import os.path as osp
import random
import time
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import ModuleList, ReLU, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.nn import BatchNorm, LEConv, global_mean_pool

from datasets import NCI1
from utils import Gtest, Gtrain, set_seed

EPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train NCI1 Model")

    parser.add_argument(
        "--data_path", nargs="?", default=osp.join(osp.dirname(__file__), "..", "data", "NCI1"), help="Input data path."
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "param", "gnns"),
        help="path for saving trained model.",
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--verbose", type=int, default=10, help="Interval of evaluation.")
    parser.add_argument("--num_unit", type=int, default=2, help="number of Convolution layers(units)")
    parser.add_argument(
        "--random_label", type=bool, default=False, help="train a model under label randomization for sanity check"
    )
    return parser.parse_args()


class NCI1GCN(torch.nn.Module):
    def __init__(self, conv_unit=3):
        super(NCI1GCN, self).__init__()
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()
        # self.convs.append(GCNConv(in_channels=37, out_channels=128))
        self.convs.append(LEConv(in_channels=37, out_channels=128))
        for i in range(conv_unit - 2):
            #     self.convs.append(GCNConv(in_channels=128, out_channels=128))
            # self.convs.append(GCNConv(in_channels=128, out_channels=128))
            self.convs.append(LEConv(in_channels=128, out_channels=128))
        self.convs.append(LEConv(in_channels=128, out_channels=128))
        self.batch_norms.extend([BatchNorm(128)] * conv_unit)
        self.relus.extend([ReLU()] * conv_unit)

        # self.lin1 = Lin(128, 128)
        self.ffn = nn.Sequential(*([nn.Linear(128, 128)] + [ReLU(), nn.Dropout(), nn.Linear(128, 2)]))

        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index, batch):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight)
            # x = relu(batch_norm(x))
            x = relu(x)
        graph_x = global_mean_pool(x, batch)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return pred

    def get_node_reps(self, x, edge_index):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight)
            # x = relu(batch_norm(x))
            x = relu(x)
        node_x = x
        return node_x

    def get_graph_rep(self, x, edge_index, batch):
        node_x = self.get_node_reps(x, edge_index)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, x, edge_index, batch):
        graph_x = self.get_graph_rep(x, edge_index, batch)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return self.readout, pred

    def get_pred_explain(self, x, edge_index, edge_mask, batch):
        edge_mask = (edge_mask * EPS).sigmoid()
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight=edge_mask)
            x = relu(x)
        node_x = x
        graph_x = global_mean_pool(node_x, batch)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return self.readout, pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


if __name__ == "__main__":
    set_seed(0)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    test_dataset = NCI1(args.data_path, mode="testing")
    val_dataset = NCI1(args.data_path, mode="evaluation")
    train_dataset = NCI1(args.data_path, mode="training")
    if args.random_label:
        for dataset in [test_dataset, val_dataset, train_dataset]:
            for g in dataset:
                g.y.fill_(random.choice([0, 1]))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = NCI1GCN(args.num_unit).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-5)
    min_error = None
    for epoch in range(1, args.epoch + 1):
        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        loss = Gtrain(train_loader, model, optimizer, device=device, criterion=nn.CrossEntropyLoss())

        _, train_acc = Gtest(train_loader, model, device=device, criterion=nn.CrossEntropyLoss())

        val_error, val_acc = Gtest(val_loader, model, device=device, criterion=nn.CrossEntropyLoss())
        test_error, test_acc = Gtest(test_loader, model, device=device, criterion=nn.CrossEntropyLoss())
        scheduler.step(val_error)
        if min_error is None or val_error <= min_error:
            min_error = val_error

        t2 = time.time()

        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader, model, device=device, criterion=nn.CrossEntropyLoss())
            t3 = time.time()
            print(
                "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, "
                "Test acc: {:.5f}".format(epoch, t3 - t1, lr, loss, test_error, test_acc)
            )
            continue

        print(
            "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, Validation Loss: {:.5f}, "
            "Validation acc: {:5f}".format(epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc)
        )

    save_path = "NCI1_gcn.pt"
    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))
