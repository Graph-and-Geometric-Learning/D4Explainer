import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset


class bbbp(InMemoryDataset):
    splits = ["training", "evaluation", "testing"]

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(bbbp, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(osp.join(self.root, "processed/data.pt"))
        self.data.x = self.data.x.float()
        edge_attr = self.data.edge_attr
        loc = torch.where(edge_attr != edge_attr)
        edge_attr[loc] = 0
        self.data.edge_attr = edge_attr.float()
        self.data.y = self.data.y.view(-1).to(torch.int64)
