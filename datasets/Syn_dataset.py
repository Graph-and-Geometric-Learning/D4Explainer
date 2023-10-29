import os
import os.path as osp
import pickle

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, k_hop_subgraph, subgraph


def get_neighbourhood(
    node_idx, edge_index, features, labels, edge_label_matrix, n_hops
):
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)  # Get all nodes involved
    edge_index_unlabel = edge_subset[1]
    ground_truth = torch.zeros((edge_index_unlabel.size(1)))
    for t, (i, j) in enumerate(zip(edge_index_unlabel[0], edge_index_unlabel[1])):
        if edge_label_matrix[i, j] == 1:
            ground_truth[t] = 1
    ground_truth = ground_truth.bool()
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
    return (
        sub_feat,
        edge_index_sub,
        sub_labels,
        self_label,
        node_dict,
        ground_truth,
        mapping_mask,
    )


class SynGraphDataset(InMemoryDataset):
    def __init__(self, root, name, mode="testing", transform=None, pre_transform=None):
        self.name = name
        self.mode = mode
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
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
        with open(
            os.path.join(f"./data/{self.name}/raw", f"{self.name}.pkl"), "rb"
        ) as f:
            (
                adj,
                features,
                y_train,
                y_val,
                y_test,
                train_mask,
                val_mask,
                test_mask,
                edge_label_matrix,
            ) = pickle.load(f)
        x = torch.from_numpy(features).float()
        y = (
            train_mask.reshape(-1, 1) * y_train
            + val_mask.reshape(-1, 1) * y_val
            + test_mask.reshape(-1, 1) * y_test
        )
        y = torch.from_numpy(np.where(y)[1])

        edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        data_whole = Data(x=x, edge_index=edge_index, y=y)
        data_whole.train_mask = torch.from_numpy(train_mask)
        data_whole.val_mask = torch.from_numpy(val_mask)
        data_whole.test_mask = torch.from_numpy(test_mask)
        torch.save(data_whole, f"./data/{self.name}/processed/whole_graph.pt")

        data_list = []
        for id in range(x.shape[0]):
            (
                sub_feat,
                edge_index_sub,
                sub_labels,
                self_label,
                node_dict,
                ground_truth,
                mapping_mask,
            ) = get_neighbourhood(
                id,
                edge_index,
                features=x,
                labels=y,
                edge_label_matrix=edge_label_matrix,
                n_hops=4,
            )
            data = Data(
                x=sub_feat,
                edge_index=edge_index_sub,
                y=sub_labels,
                self_y=self_label,
                node_dict=node_dict,
                ground_truth=ground_truth,
                mapping=mapping_mask,
                idx=id,
            )
            print(data)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        # data_list = np.array(data_list)
        train_mask = list(np.where(train_mask)[0])
        val_mask = list(np.where(val_mask)[0])
        test_mask = list(np.where(test_mask)[0])
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
