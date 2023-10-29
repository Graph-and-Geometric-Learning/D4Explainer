import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import WebKB
from torch_geometric.utils import k_hop_subgraph, subgraph


def get_neighbourhood(node_idx, edge_index, features, labels, n_hops):
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)  # Get all nodes involved
    edge_subset_relabel = subgraph(edge_subset[0], edge_index, relabel_nodes=True)
    edge_index_sub = edge_subset_relabel[0]  # [2, edge_num_sub]
    sub_feat = features[edge_subset[0], :]  # [node_num_sub, feature_dim]
    sub_labels = labels[edge_subset[0]]
    self_label = labels[node_idx]
    node_dict = torch.tensor(edge_subset[0]).reshape(-1, 1)  # Maps orig labels to new
    mapping = edge_subset[2]
    mapping_mask = torch.zeros((sub_feat.shape[0]))
    mapping_mask[mapping] = 1
    mapping_mask = mapping_mask.bool()
    return sub_feat, edge_index_sub, sub_labels, self_label, node_dict, mapping_mask


class WebDataset(InMemoryDataset):
    def __init__(self, root, name, mode="testing", transform=None, pre_transform=None):
        self.name = name
        self.mode = mode
        super(WebDataset, self).__init__(root, transform, pre_transform)
        idx = self.processed_file_names.index("{}_sub.pt".format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ["training_sub.pt", "evaluating_sub.pt", "testing_sub.pt"]

    def process(self):
        # Read data into huge `Data` list.
        webdata = WebKB(root=self.root, name=self.name)
        data = webdata[0]
        train_mask_data = data.train_mask[:, 0]
        val_mask_data = data.val_mask[:, 0]
        test_mask_data = data.test_mask[:, 0]
        edge_index = data.edge_index
        x = data.x
        y = data.y
        data_whole = Data(x=x, edge_index=edge_index, y=y)
        data_whole.train_mask = train_mask_data
        data_whole.val_mask = val_mask_data
        data_whole.test_mask = test_mask_data
        torch.save(data_whole, f"./data/{self.name}/processed/whole_graph.pt")

        data_list = []
        for id in range(x.shape[0]):
            (
                sub_feat,
                edge_index_sub,
                sub_labels,
                self_label,
                node_dict,
                mapping_mask,
            ) = get_neighbourhood(id, edge_index, features=x, labels=y, n_hops=6)
            data = Data(
                x=sub_feat,
                edge_index=edge_index_sub,
                y=sub_labels,
                self_y=self_label,
                node_dict=node_dict,
                mapping=mapping_mask,
                idx=id,
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        train_mask = list(np.where(train_mask_data)[0])
        val_mask = list(np.where(val_mask_data)[0])
        test_mask = list(np.where(test_mask_data)[0])
        torch.save(
            self.collate([data_list[i] for i in train_mask]),
            f"./data/{self.name}/processed/training_sub.pt",
        )
        torch.save(
            self.collate([data_list[i] for i in val_mask]),
            f"./data/{self.name}/processed/evaluating_sub.pt",
        )
        torch.save(
            self.collate([data_list[i] for i in test_mask]),
            f"./data/{self.name}/processed/testing_sub.pt",
        )
