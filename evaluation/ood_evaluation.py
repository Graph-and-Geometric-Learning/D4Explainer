import argparse
import os
import sys
sys.path.append("..")
import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from constants import dataset_choices, feature_dict, task_type
from evaluation.in_distribution.ood_stat import eval_graph_list
from explainers import DiffExplainer
from gnns import *
from utils.dataset import get_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="in-distribution evaluation")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--root", type=str, default="../results", help="Result directory.")
    parser.add_argument("--dataset", type=str, default="NCI1", choices=dataset_choices)
    parser.add_argument("--gnn_type", type=str, default="gcn")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--normalization", type=str, default="instance")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--layers_per_conv", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--cat_output", type=bool, default=True)
    parser.add_argument("--residual", type=bool, default=False)
    parser.add_argument("--noise_mlp", type=bool, default=True)
    parser.add_argument("--simplified", type=bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.001)
    parser.add_argument("--prob_low", type=float, default=0.0)
    parser.add_argument("--prob_high", type=float, default=0.4)
    parser.add_argument("--sigma_length", type=int, default=10)
    return parser.parse_args()


args = parse_args()
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
mr = [0.2]
args.noise_list = None
args.feature_in = feature_dict[args.dataset]
args.task = task_type[args.dataset]
train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset, root="../data/")
test_loader = DataLoader(test_dataset[: args.num_test], batch_size=1, shuffle=False, drop_last=False)
gnn_path = f"../param/gnns/{args.dataset}_{args.gnn_type}.pt"
Explainer = DiffExplainer(args.device, gnn_path)
test_graph = []
pred_graph = []
for graph in test_loader:
    graph.to(args.device)
    exp_subgraph,_,_,_ = Explainer.explain_evaluation(args, graph)
    G_ori = to_networkx(graph, to_undirected=True)
    G_pred = to_networkx(exp_subgraph, to_undirected=True)
    test_graph.append(G_ori)
    pred_graph.append(G_pred)
MMD = eval_graph_list(test_graph, pred_graph)
print(MMD)
