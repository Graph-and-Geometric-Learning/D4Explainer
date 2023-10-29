"""
Modified based on torch_geometric.nn.models.GNNExplainer
which generates explainations in node prediction tasks.

Citation:
Ying et al. GNNExplainer: Generating Explanations for Graph Neural Networks.
"""

from math import sqrt

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

EPS = 1e-15


class MetaGNNGExplainer(torch.nn.Module):
    coeffs = {
        "edge_size": 0.05,
        "edge_ent": 0.5,
    }

    def __init__(self, model, epochs=100, lr=0.01, log=True, task="gc"):
        super(MetaGNNGExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.task = task

    def __set_masks__(self, x, edge_index, init="normal"):
        N = x.size(0)
        E = edge_index.size(1)

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def __loss__(self, log_logits, pred_label):
        criterion = torch.nn.NLLLoss()
        loss = criterion(log_logits, pred_label)
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs["edge_size"] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs["edge_ent"] * ent.mean()
        return loss

    def explain_graph(self, graph, **kwargs):
        """
        Explain a graph using the MetaGNNGExplainer.
        :param graph: the graph to be explained.
        :param kwargs: additional arguments.
        :return: the explanation (edge_mask)
        """
        self.__clear_masks__()

        # get the initial prediction.
        with torch.no_grad():
            if self.task == "nc":
                soft_pred, _ = self.model.get_node_pred_subgraph(
                    x=graph.x, edge_index=graph.edge_index, mapping=graph.mapping
                )
            else:
                soft_pred, _ = self.model.get_pred(x=graph.x, edge_index=graph.edge_index, batch=graph.batch)
            pred_label = soft_pred.argmax(dim=-1)

        N = graph.x.size(0)
        E = graph.edge_index.size(1)

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std).to()
        self.to(graph.x.device)
        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            if self.task == "nc":
                _, output_repr = self.model.get_pred_explain(
                    x=graph.x, edge_index=graph.edge_index, edge_mask=self.edge_mask, mapping=graph.mapping
                )
            else:
                _, output_repr = self.model.get_pred_explain(
                    x=graph.x, edge_index=graph.edge_index, edge_mask=self.edge_mask, batch=graph.batch
                )
            log_logits = F.log_softmax(output_repr)
            loss = self.__loss__(log_logits, pred_label)
            loss.backward()
            optimizer.step()

        edge_mask = self.edge_mask.detach().sigmoid()

        return edge_mask

    def __repr__(self):
        return f"{self.__class__.__name__}()"
