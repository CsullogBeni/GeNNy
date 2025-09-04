"""
Abstract base class and utilities for learning over AST graphs.

Key ideas:
- Node attributes are (class_, value). We encode both as categorical IDs
  (via LabelEncoder) and project them with small MLPs to `hidden_dim`,
  then concatenate → [N, 2*hidden_dim].
- The GNN backbone is a small GCN (see `graph_learner.gnn.GNN`).
- Data augmentation during training: per-epoch random pruning.
  Early epochs remove leaves; later epochs remove entire subtrees.
- Subclasses define the task specifics (labels, training loop, prediction).
"""

import json
import random
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

from graph_learner.gnn import GNN


class AbstractGraphLearner(nn.Module, ABC):
    """
    Abstract base for AST graph learning tasks.

    This class encapsulates:
    - robust categorical feature encoding for `(class_, value)` pairs,
    - a shared GNN encoder (GCN),
    - utilities for reading JSON graphs and converting them to PyG `Data`,
    - progressive, per-epoch structural pruning,
    - a training harness that delegates the actual optimization to subclasses.

    Subclassing:
        Implement:
          * `_label_nodes(self, graph: nx.DiGraph) -> torch.Tensor`
          * `_train_epoch(self, dataset: List[Data], epoch: int) -> None`
          * `predict_subgraph(self, graph_path: str, node_embeddings: torch.Tensor) -> List[int]`
    """

    def __init__(self, hidden_dim: int = 64, device: str = "cpu",
                 gnn_layers: int = 2, gnn_dropout: float = 0.0):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)

        self.device = device
        self.hidden_dim = hidden_dim

        # Sentinel tokens for robust encoding
        self.CLASS_UNK = "<UNK_CLASS>"
        self.VALUE_PAD = "<PAD>"
        self.VALUE_NONE = "<NONE>"
        self.VALUE_UNK = "<UNK_VALUE>"

        # Categorical encoders
        self.class_encoder = LabelEncoder()
        self.value_encoder = LabelEncoder()

        # Small projection MLPs ("autoencoders") for integer IDs → hidden_dim
        self.class_autoencoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.value_autoencoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GNN: input is concat([class_embed, value_embed]) = 2*hidden_dim
        self.gnn = GNN(hidden_dim * 2, hidden_dim, num_layers=gnn_layers, dropout=gnn_dropout)

        # Single optimizer over encoders + GNN; subclasses may extend/replace
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Move all parameters & default-allocated tensors to device
        self.to(self.device)

    # -----------------------------------------------------------
    # ENCODING
    # -----------------------------------------------------------
    @staticmethod
    def _ensure_unk_in_encoder(encoder: LabelEncoder, unk_token: str) -> None:
        """
        Ensure an UNK token exists in a fitted LabelEncoder.

        Args:
            encoder (LabelEncoder): A fitted scikit-learn LabelEncoder. If it is not
                fitted yet (no `classes_`), the method returns silently.
            unk_token (str): The token to add as "unknown" class.
        """
        if not hasattr(encoder, "classes_"):
            return
        classes = encoder.classes_
        classes_list = classes.tolist() if hasattr(classes, "tolist") else list(classes)
        if unk_token not in classes_list:
            encoder.classes_ = np.array(classes_list + [unk_token], dtype=object)
            print(f"[enc] injected UNK token '{unk_token}' → vocab size = {len(encoder.classes_)}")

    def _safe_transform(self, encoder: LabelEncoder, values: List[str], unk_token: str):
        """
        Transform with a LabelEncoder; unseen values are mapped to `unk_token`.

        Args:
            encoder (LabelEncoder): A fitted LabelEncoder.
            values (List[str]): Sequence of tokens to transform.
            unk_token (str): Token used to represent unknowns.

        Returns:
            numpy.ndarray: Encoded integer IDs for each input value.
        """
        self._ensure_unk_in_encoder(encoder, unk_token)
        classes = encoder.classes_
        classes_set = set(classes.tolist() if hasattr(classes, "tolist") else list(classes))
        mapped = [v if v in classes_set else unk_token for v in values]
        return encoder.transform(mapped)

    def fit_encoders(self, filepaths: List[str]) -> None:
        """
        Fit the `(class_, value)` LabelEncoders over all graphs.

        Args:
            filepaths (List[str]): JSON graph filepaths. Each file must contain
                {"nodes":[...], "edges":[...]} where nodes have at least "id" and "class_",
                and optionally "value".

        Notes:
            - Guarantees sentinel tokens are present:
              class → `<UNK_CLASS>`, value → `<PAD>`, `<NONE>`, `<UNK_VALUE>`.
            - Prints vocabulary sizes after fitting.
        """
        class_values, value_values = [], []

        for path in filepaths:
            g = self._load_graph_from_json(path)
            for _, attrs in g.nodes(data=True):
                c = attrs.get("class_", self.CLASS_UNK)
                class_values.append(str(c))
                if attrs.get("class_") == "TerminalNodeImpl":
                    v = attrs.get("value")
                    v = self.VALUE_NONE if v is None else str(v)
                else:
                    v = self.VALUE_PAD
                value_values.append(v)

        # Ensure sentinel tokens are part of the vocab
        class_values += [self.CLASS_UNK]
        value_values += [self.VALUE_PAD, self.VALUE_NONE, self.VALUE_UNK]

        self.class_encoder.fit(class_values)
        self.value_encoder.fit(value_values)

        print(f"[enc] class_vocab={len(getattr(self.class_encoder, 'classes_', []))} "
              f"value_vocab={len(getattr(self.value_encoder, 'classes_', []))}")

    def _encode_node_features(self, node_attrs: List[Dict]) -> torch.Tensor:
        """
        Encode a list of node attribute dicts into concatenated embeddings.

        Args:
            node_attrs (List[Dict]): Per-node attribute dicts. Expected keys:
                - "class_" (str): node type (required).
                - "value" (any): optional; used only if class_ == "TerminalNodeImpl".
                  If None in that case, it maps to `<NONE>`. Non-terminals use `<PAD>`.

        Returns:
            torch.Tensor: Node features of shape [N, 2*hidden_dim] on `self.device`.
                          First half is class embedding; second half is value embedding.
        """
        class_values, value_values = [], []

        for n in node_attrs:
            c = str(n.get("class_", self.CLASS_UNK))
            class_values.append(c)

            if n.get("class_") == "TerminalNodeImpl":
                v = n.get("value")
                v = self.VALUE_NONE if v is None else str(v)
            else:
                v = self.VALUE_PAD
            value_values.append(v)

        class_ids = torch.as_tensor(
            self._safe_transform(self.class_encoder, class_values, self.CLASS_UNK),
            dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        value_ids = torch.as_tensor(
            self._safe_transform(self.value_encoder, value_values, self.VALUE_UNK),
            dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        class_embed = self.class_autoencoder(class_ids)
        value_embed = self.value_autoencoder(value_ids)
        x = torch.cat([class_embed, value_embed], dim=1)

        print(f"[encode] nodes={len(node_attrs)} → x.shape={tuple(x.shape)}")
        return x

    # -----------------------------------------------------------
    # GRAPH I/O & CONVERSION
    # -----------------------------------------------------------
    @staticmethod
    def _load_graph_from_json(filepath: str) -> nx.DiGraph:
        """
        Load a directed graph from a JSON file.

        Args:
            filepath (str): Path to a JSON file with structure:
                {
                  "nodes": [{"id": ..., "class_": ..., "value": ...}, ...],
                  "edges": [{"source": ..., "target": ...}, ...]
                }

        Returns:
            nx.DiGraph: Directed graph with node attributes copied verbatim.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        g = nx.DiGraph()
        for node in data.get("nodes", []):
            g.add_node(node["id"], **node)

        for edge in data.get("edges", []):
            src = edge.get("source")
            dst = edge.get("target")
            if src is not None and dst is not None and src in g and dst in g:
                g.add_edge(src, dst)

        print(f"[io] loaded '{filepath}': |V|={g.number_of_nodes()} |E|={g.number_of_edges()}")
        return g

    def _graph_to_pyg(self, graph: nx.DiGraph) -> Data:
        """
        Convert a NetworkX directed graph to a PyG `Data`.

        Args:
            graph (nx.DiGraph): Input graph. Nodes should include "class_" and optionally "value".

        Returns:
            torch_geometric.data.Data: PyG data with:
                - x: [N, 2*hidden_dim] node features on `self.device`
                - edge_index: [2, E] long tensor on `self.device`
                - _raw_node_attrs: list of node-attr dicts (for re-encoding if needed)
                - _node_ids: list of node IDs in the exact order used for x/edges
        """
        nodes = list(graph.nodes(data=True))
        if len(nodes) == 0:
            graph.add_node("__DUMMY__", class_="<DUMMY>", value=self.VALUE_PAD)
            nodes = list(graph.nodes(data=True))

        node_ids = [nid for nid, _ in nodes]
        node_attrs = [attr for _, attr in nodes]
        x = self._encode_node_features(node_attrs)

        id_map = {nid: idx for idx, nid in enumerate(node_ids)}

        # Original directed edges
        e = [[id_map[src], id_map[dst]] for src, dst in graph.edges
             if src in id_map and dst in id_map]
        # Bi-directional augmentation + self-loops (stabilizes GCN propagation)
        e = e + [[b, a] for a, b in e] + [[i, i] for i in range(len(node_ids))]

        if len(e) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            edge_index = torch.as_tensor(e, dtype=torch.long, device=self.device).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        data._raw_node_attrs = node_attrs
        data._node_ids = node_ids

        print(f"[to_pyg] |V|={len(node_ids)} |E|={edge_index.shape[1]} (after sym+loops)")
        return data

    # -----------------------------------------------------------
    # PRUNING: LEAVES / SUBTREES
    # -----------------------------------------------------------
    @staticmethod
    def _delete_random_leaves(graph: nx.DiGraph, ratio: float) -> nx.DiGraph:
        """
        Randomly delete leaf nodes (out_degree == 0).

        Args:
            graph (nx.DiGraph): Graph to be modified in-place.
            ratio (float): Fraction of total nodes to remove (bounded by leaves).
                           Deletes at least 1 node if any leaves exist.

        Returns:
            nx.DiGraph: The same graph instance, pruned in-place.
        """
        if len(graph) == 0:
            return graph

        leaves = [n for n in graph.nodes if graph.out_degree(n) == 0]
        if not leaves:
            return graph

        num_delete = max(1, int(len(graph) * ratio))
        delete_nodes = random.sample(leaves, min(len(leaves), num_delete))
        graph.remove_nodes_from(delete_nodes)
        return graph

    @staticmethod
    def _delete_random_subtrees(graph: nx.DiGraph, ratio: float) -> nx.DiGraph:
        """
        Randomly delete entire subtrees rooted at internal nodes.

        Args:
            graph (nx.DiGraph): Graph to be modified in-place.
            ratio (float): Target fraction of total nodes to delete. Internal
                           roots are sampled; each root and all its descendants
                           are removed until the target is reached.

        Returns:
            nx.DiGraph: The same graph instance, pruned in-place.
        """
        if len(graph) == 0:
            return graph

        target_delete = max(1, int(len(graph) * ratio))
        candidates = [n for n in graph.nodes if graph.out_degree(n) > 0]
        random.shuffle(candidates)

        to_delete = set()
        for c in candidates:
            if len(to_delete) >= target_delete:
                break
            sub = {c}
            try:
                sub |= nx.descendants(graph, c)
            except nx.NetworkXError:
                pass
            for node in sub:
                to_delete.add(node)
                if len(to_delete) >= target_delete:
                    break

        # Keep at least one node
        if len(to_delete) >= len(graph):
            keep_one = random.choice(list(graph.nodes))
            to_delete.discard(keep_one)

        graph.remove_nodes_from(list(to_delete))
        return graph

    # -----------------------------------------------------------
    # LABEL PREPARATION
    # -----------------------------------------------------------
    def _get_label_tensor(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        Delegate label generation to the subclass' `_label_nodes`.

        Args:
            graph (nx.DiGraph): The graph for which labels are required.

        Returns:
            torch.Tensor: Float tensor of shape [N], aligned with `_graph_to_pyg` order.
        """
        return self._label_nodes(graph)

    @abstractmethod
    def _label_nodes(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        Compute node-level labels for a given graph.

        Args:
            graph (nx.DiGraph): Input graph.

        Returns:
            torch.Tensor: Labels as a float tensor of shape [N], aligned with the node
                          order used by `_graph_to_pyg`.
        """
        raise NotImplementedError

    # -----------------------------------------------------------
    # TRAIN HARNESS
    # -----------------------------------------------------------
    def fit(self, filepaths: List[str], epochs: int = 10,
            leaf_phase_epochs: int = 5, prune_step: float = 0.05,
            prune_max_ratio: float = 0.30) -> None:
        """
        Train-time augmentation with progressive pruning.

        Args:
            filepaths (List[str]): JSON graph filepaths to iterate over each epoch.
            epochs (int): Number of epochs (must be > 0).
            leaf_phase_epochs (int): For the first `leaf_phase_epochs`, only leaf
                                     deletion is applied. Afterward, subtree deletion.
            prune_step (float): Increment in deletion ratio per epoch.
            prune_max_ratio (float): Upper bound on deletion ratio within an epoch.

        Notes:
            For epoch t (0-based): ratio = min(prune_step * (t + 1), prune_max_ratio).
            Each filepath is loaded, pruned, converted to PyG `Data`, then passed
            to `_train_epoch`.
        """
        assert epochs > 0
        self.fit_encoders(filepaths)
        self.train()

        for epoch in range(epochs):
            ratio = min(prune_step * (epoch + 1), prune_max_ratio)
            all_data: List[Data] = []

            for path in filepaths:
                g = self._load_graph_from_json(path)
                if epoch > 0:
                    if epoch < leaf_phase_epochs:
                        g = self._delete_random_leaves(g, ratio=ratio)
                    else:
                        g = self._delete_random_subtrees(g, ratio=ratio)

                data = self._graph_to_pyg(g)
                all_data.append(data)

            print(f"[train] epoch={epoch} prune_ratio={ratio:.3f} graphs={len(all_data)}")
            self._train_epoch(all_data, epoch)

    # -----------------------------------------------------------
    # INFERENCE
    # -----------------------------------------------------------
    def encode_graph(self, filepath: str) -> torch.Tensor:
        """
        Compute node embeddings for a single JSON graph.

        Args:
            filepath (str): Path to the JSON graph.

        Returns:
            torch.Tensor: Node embeddings of shape [N, hidden_dim], produced by the GNN.
        """
        self.eval()
        with torch.no_grad():
            g = self._load_graph_from_json(filepath)
            data = self._graph_to_pyg(g)
            out = self.gnn(data.x, data.edge_index)
        print(f"[infer] '{filepath}' → emb.shape={tuple(out.shape)}")
        return out

    # -----------------------------------------------------------
    # ABSTRACT: EPOCH & PREDICTION
    # -----------------------------------------------------------
    @abstractmethod
    def _train_epoch(self, dataset: List[Data], epoch: int) -> None:
        """
        Run one training epoch over a list of PyG `Data` graphs.

        Args:
            dataset (List[Data]): Per-graph mini-batches prepared by `_graph_to_pyg`.
            epoch (int): Current epoch index (0-based), provided for scheduling/prints.
        """
        pass

    @abstractmethod
    def predict_subgraph(self, graph_path: str, node_embeddings: torch.Tensor) -> List[int]:
        """
        Convert node embeddings into predicted node indices for a given task.

        Args:
            graph_path (str): Path to the JSON graph used for prediction.
            node_embeddings (torch.Tensor): GNN outputs of shape [N, hidden_dim].

        Returns:
            List[int]: Predicted node indices (0-based, aligned with `_graph_to_pyg` order).
        """
        pass
