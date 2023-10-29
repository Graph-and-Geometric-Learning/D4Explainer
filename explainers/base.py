import io
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import cm
from PIL import Image

from .visual import graph_to_mol, vis_dict

EPS = 1e-6


class Explainer(object):
    def __init__(self, device, gnn_model_path, task="gc"):
        self.device = device
        self.model = torch.load(gnn_model_path, map_location=self.device).to(self.device)
        self.model.eval()
        self.model_name = self.model.__class__.__name__
        self.name = self.__class__.__name__

        self.path = gnn_model_path
        self.last_result = None
        self.vis_dict = None
        self.task = task

    def explain_graph(self, graph, **kwargs):
        """
        Main part for different graph attribution methods
        :param graph: target graph instance to be explained
        :param kwargs: other parameters
        :return: edge_imp, i.e., attributions for edges, which are derived from the attribution methods.
        """
        raise NotImplementedError

    @staticmethod
    def get_rank(lst, r=1):
        topk_idx = list(np.argsort(-lst))
        top_pred = np.zeros_like(lst)
        n = len(lst)
        k = int(r * n)
        for i in range(k):
            top_pred[topk_idx[i]] = n - i
        return top_pred

    @staticmethod
    def norm_imp(imp):
        imp[imp < 0] = 0
        imp += 1e-16
        return imp / imp.sum()

    def __relabel__(self, g, edge_index):
        sub_nodes = torch.unique(edge_index)
        x = g.x[sub_nodes]
        batch = g.batch[sub_nodes]
        row, col = edge_index
        pos = None
        try:
            pos = g.pos[sub_nodes]
        except Exception:
            pass

        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((g.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
        return x, edge_index, batch, pos

    def __reparameterize__(self, log_alpha, beta=0.1, training=True):
        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def pack_explanatory_subgraph(self, top_ratio=0.2, graph=None, imp=None, relabel=False, if_cf=False):
        """
        Pack the explanatory subgraph from the original graph
        :param top_ratio: the ratio of edges to be selected
        :param graph: the original graph
        :param imp: the attribution scores for edges
        :param relabel: whether to relabel the nodes in the explanatory subgraph
        :param if_cf: whether to use the CF method
        :return: the explanatory subgraph
        """
        if graph is None:
            graph, imp = self.last_result
        assert len(imp) == graph.num_edges, "length mismatch"

        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        exp_subgraph.y = graph.y if self.task == "gc" else graph.self_y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = min(max(math.ceil(top_ratio * Gi_n_edge), 1), Gi_n_edge)
            if not if_cf:
                Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            else:
                Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[topk:]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        try:
            exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        except Exception:
            pass
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]

        exp_subgraph.x = graph.x
        if relabel:
            (exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, exp_subgraph.pos) = self.__relabel__(
                exp_subgraph, exp_subgraph.edge_index
            )
        return exp_subgraph

    def evaluate_acc(self, top_ratio_list, graph=None, imp=None, if_cf=False):
        """
        Evaluate the accuracy of the explanatory subgraph
        :param top_ratio_list: the ratio of edges to be selected
        :param graph: the original graph
        :param imp: the attribution scores for edges
        :param if_cf: whether to generate cf explanation
        :return: the accuracy of the explanatory subgraph
        """
        if graph is None:
            assert self.last_result is not None
            graph, imp = self.last_result
        acc = np.array([[]])
        fidelity = np.array([[]])
        if self.task == "nc":
            output_prob, _ = self.model.get_node_pred_subgraph(
                x=graph.x, edge_index=graph.edge_index, mapping=graph.mapping
            )
        else:
            output_prob, _ = self.model.get_pred(x=graph.x, edge_index=graph.edge_index, batch=graph.batch)
        y_pred = output_prob.argmax(dim=-1)
        for idx, top_ratio in enumerate(top_ratio_list):
            exp_subgraph = self.pack_explanatory_subgraph(top_ratio, graph=graph, imp=imp, if_cf=if_cf)
            if self.task == "nc":
                soft_pred, _ = self.model.get_node_pred_subgraph(
                    x=exp_subgraph.x, edge_index=exp_subgraph.edge_index, mapping=exp_subgraph.mapping
                )
            else:
                soft_pred, _ = self.model.get_pred(
                    x=exp_subgraph.x, edge_index=exp_subgraph.edge_index, batch=exp_subgraph.batch
                )
            # soft_pred: [bsz, num_class]
            res_acc = (y_pred == soft_pred.argmax(dim=-1)).detach().cpu().float().view(-1, 1).numpy()
            labels = torch.LongTensor([[i] for i in y_pred]).to(y_pred.device)
            if not if_cf:
                res_fid = soft_pred.gather(1, labels).detach().cpu().float().view(-1, 1).numpy()
            else:
                res_fid = (1 - soft_pred.gather(1, labels)).detach().cpu().float().view(-1, 1).numpy()
            acc = np.concatenate([acc, res_acc], axis=1)  # [bsz, len_ratio_list]
            fidelity = np.concatenate([fidelity, res_fid], axis=1)
        return acc, fidelity

    def visualize(
        self, graph=None, edge_imp=None, counter_edge_index=None, vis_ratio=0.2, save=False, layout=False, name=None
    ):
        """
        Visualize the attribution scores for edges (xx-Motif / Mutag)
        # TODO: visualization for BBBP / node classification
        :param graph: the original graph
        :param edge_imp: the attribution scores for edges
        :param counter_edge_index: the counterfactual edges
        :param vis_ratio: the ratio of edges to be visualized
        :param save: whether to save the visualization
        :param layout: whether to use the layout
        :param name: the name of the visualization
        :return: None
        """
        if graph is None:
            assert self.last_result is not None
            graph, edge_imp = self.last_result

        topk = max(int(vis_ratio * graph.num_edges), 1)
        idx = np.argsort(-edge_imp)[:topk]
        G = nx.DiGraph()
        G.add_nodes_from(range(graph.num_nodes))
        G.add_edges_from(list(graph.edge_index.cpu().numpy().T))

        if counter_edge_index is not None:
            G.add_edges_from(list(counter_edge_index.cpu().numpy().T))
        if self.vis_dict is None:
            self.vis_dict = vis_dict[self.model_name] if self.model_name in vis_dict.keys() else vis_dict["default"]

        folder = Path(r"image/%s" % (self.model_name))
        if save and not os.path.exists(folder):
            os.makedirs(folder)

        edge_pos_mask = np.zeros(graph.num_edges, dtype=np.bool_)
        edge_pos_mask[idx] = True
        vmax = sum(edge_pos_mask)
        node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
        node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
        node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
        node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
        node_pos_mask[node_pos_idx] = True
        node_neg_mask[node_neg_idx] = True

        if "Motif" in self.model_name:
            plt.figure(figsize=(8, 6), dpi=100)
            pos = graph.pos[0]
            nx.draw_networkx_nodes(
                G,
                pos={i: pos[i] for i in node_pos_idx},
                nodelist=node_pos_idx,
                node_size=self.vis_dict["node_size"],
                node_color=graph.z[0][node_pos_idx],
                alpha=1,
                cmap="winter",
                linewidths=self.vis_dict["linewidths"],
                edgecolors="red",
                vmin=-max(graph.z[0]),
                vmax=max(graph.z[0]),
            )
            nx.draw_networkx_nodes(
                G,
                pos={i: pos[i] for i in node_neg_idx},
                nodelist=node_neg_idx,
                node_size=self.vis_dict["node_size"],
                node_color=graph.z[0][node_neg_idx],
                alpha=0.2,
                cmap="winter",
                linewidths=self.vis_dict["linewidths"],
                edgecolors="whitesmoke",
                vmin=-max(graph.z[0]),
                vmax=max(graph.z[0]),
            )
            nx.draw_networkx_edges(
                G,
                pos=pos,
                edgelist=list(graph.edge_index.cpu().numpy().T),
                edge_color="whitesmoke",
                width=self.vis_dict["width"],
                arrows=False,
            )
            nx.draw_networkx_edges(
                G,
                pos=pos,
                edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                edge_color=self.get_rank(edge_imp[edge_pos_mask]),
                # np.ones(len(edge_imp[edge_pos_mask])),
                width=self.vis_dict["width"],
                edge_cmap=cm.get_cmap("bwr"),
                edge_vmin=-vmax,
                edge_vmax=vmax,
                arrows=False,
            )
            if counter_edge_index is not None:
                nx.draw_networkx_edges(
                    G,
                    pos=pos,
                    edgelist=list(counter_edge_index.cpu().numpy().T),
                    edge_color="mediumturquoise",
                    width=self.vis_dict["width"] / 3.0,
                    arrows=False,
                )

        if "Mutag" in self.model_name:
            from rdkit.Chem.Draw import rdMolDraw2D

            idx = [int(i / 2) for i in idx]
            x = graph.x.detach().cpu().tolist()
            edge_index = graph.edge_index.T.detach().cpu().tolist()
            edge_attr = graph.edge_attr.detach().cpu().tolist()
            mol = graph_to_mol(x, edge_index, edge_attr)
            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            hit_at = np.unique(graph.edge_index[:, idx].detach().cpu().numpy()).tolist()

            def add_atom_index(mol):
                atoms = mol.GetNumAtoms()
                for i in range(atoms):
                    mol.GetAtomWithIdx(i).SetProp("molAtomMapNumber", str(mol.GetAtomWithIdx(i).GetIdx()))
                return mol

            hit_bonds = []
            for u, v in graph.edge_index.T[idx]:
                hit_bonds.append(mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx())
            rdMolDraw2D.PrepareAndDrawMolecule(
                d,
                mol,
                highlightAtoms=hit_at,
                highlightBonds=hit_bonds,
                highlightAtomColors={i: (0, 1, 0) for i in hit_at},
                highlightBondColors={i: (0, 1, 0) for i in hit_bonds},
            )
            d.FinishDrawing()
            bindata = d.GetDrawingText()
            iobuf = io.BytesIO(bindata)
            image = Image.open(iobuf)
            image.show()
            if save:
                if name:
                    d.WriteDrawingText("image/%s/%s-%d-%s.png" % (self.model_name, name, int(graph.y[0]), self.name))
                else:
                    d.WriteDrawingText(
                        "image/%s/%s-%d-%s.png" % (self.model_name, str(graph.name[0]), int(graph.y[0]), self.name)
                    )
