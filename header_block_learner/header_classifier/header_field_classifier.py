import os
import pickle
from typing import List

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from genny.abs_genny import AbsGenny


class HeaderFieldClassifier(AbsGenny):
    """
    A graph-based neural classifier for detecting field nodes inside header
    declarations of abstract syntax graphs.

    This model extends `AbsGenny` and focuses on identifying
    `StructFieldContext` nodes that occur under `HeaderTypeDeclarationContext`.
    It leverages a deeper GNN (multiple layers + dropout) to capture
    multi-hop dependencies inside header subtrees.

    Key features:
        - Encodes AST nodes into vector representations.
        - Trains a multi-layer GNN to propagate structural information.
        - Uses a binary classification head for identifying struct fields.
        - Restricts training to nodes within header subtrees to reduce noise.

    Training process:
        1. Encoders are fitted on input JSON graphs.
        2. Graphs are pruned (removing leaves or subtrees) to improve robustness.
        3. Labels are assigned: only `StructFieldContext` nodes inside headers
           are marked positive.
        4. A subtree mask is applied so that only header-related nodes (positive
           or negative) are used for training.
        5. Model is trained with a weighted binary cross-entropy loss to
           handle class imbalance.

    Saving and loading:
        - The model can be saved together with optimizer state and encoders.
        - A trained model can be restored for inference or retraining.

    Args:
        hidden_dim (int, optional): Size of the hidden representation in the GNN. Default is 64.
        device (str, optional): Device to run the model on, e.g. `"cpu"` or `"cuda"`. Default is `"cpu"`.
        gnn_layers (int, optional): Number of GNN layers. Default is 5.
        gnn_dropout (float, optional): Dropout rate applied in GNN layers. Default is 0.10.
    """

    def __init__(self, hidden_dim: int = 64, device: str = "cpu",
                 gnn_layers: int = 5, gnn_dropout: float = 0.10):
        super().__init__(hidden_dim=hidden_dim, device=device, gnn_layers=gnn_layers, gnn_dropout=gnn_dropout)
        self.head = nn.Linear(self.hidden_dim, 1)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def _label_nodes(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        Generate binary labels for nodes in the input graph.

        A node is labeled as positive (1.0) if it is a `StructFieldContext`
        that belongs to the subtree of a `HeaderTypeDeclarationContext`.
        All other nodes receive a label of 0.0.

        Args:
            graph (nx.DiGraph): Input directed graph with node attributes.

        Returns:
            torch.Tensor: A tensor of shape `[num_nodes]` with float labels.
        """
        labels = torch.zeros(len(graph.nodes), dtype=torch.float32, device=self.device)
        id_to_index = {nid: i for i, nid in enumerate(graph.nodes)}

        for node_id in graph.nodes:
            if graph.nodes[node_id].get("class_") != "InputContext":
                continue
            for c1 in graph.successors(node_id):
                if graph.nodes[c1].get("class_") != "DeclarationContext": continue
                for c2 in graph.successors(c1):
                    if graph.nodes[c2].get("class_") != "TypeDeclarationContext": continue
                    for c3 in graph.successors(c2):
                        if graph.nodes[c3].get("class_") != "DerivedTypeDeclarationContext": continue
                        for c4 in graph.successors(c3):
                            if graph.nodes[c4].get("class_") != "HeaderTypeDeclarationContext": continue
                            # BFS traversal of the header subtree
                            q, vis = [c4], set()
                            while q:
                                cur = q.pop(0)
                                if cur in vis: continue
                                vis.add(cur)
                                if graph.nodes[cur].get("class_") == "StructFieldContext":
                                    labels[id_to_index[cur]] = 1.0
                                q.extend(list(graph.successors(cur)))
        return labels

    @staticmethod
    def _header_subtree_mask(graph: nx.DiGraph) -> set:
        """
        Identify all node IDs that belong to header subtrees.

        This mask is used to restrict training to nodes within header
        declarations (both positives and negatives).

        Args:
            graph (nx.DiGraph): Input directed graph with node attributes.

        Returns:
            set: A set of node IDs belonging to header subtrees.
        """
        mask_ids = set()
        for node_id in graph.nodes:
            if graph.nodes[node_id].get("class_") != "InputContext":
                continue
            for c1 in graph.successors(node_id):
                if graph.nodes[c1].get("class_") != "DeclarationContext": continue
                for c2 in graph.successors(c1):
                    if graph.nodes[c2].get("class_") != "TypeDeclarationContext": continue
                    for c3 in graph.successors(c2):
                        if graph.nodes[c3].get("class_") != "DerivedTypeDeclarationContext": continue
                        for c4 in graph.successors(c3):
                            if graph.nodes[c4].get("class_") != "HeaderTypeDeclarationContext":
                                continue
                            # BFS traversal to collect entire header subtree
                            q, vis = [c4], set([c4])
                            while q:
                                cur = q.pop(0)
                                mask_ids.add(cur)
                                for nxt in graph.successors(cur):
                                    if nxt not in vis:
                                        vis.add(nxt)
                                        q.append(nxt)
        return mask_ids

    def fit(self, filepaths: List[str], epochs: int = 20,
            leaf_phase_epochs: int = 5, prune_step: float = 0.05, prune_max_ratio: float = 0.30):
        """
        Train the classifier on a dataset of JSON graphs.

        Training uses two phases of pruning:
            - Early epochs: random leaf deletion.
            - Later epochs: random subtree deletion.

        Additionally, a mask is applied to ensure only header-subtree nodes
        contribute to training.

        Args:
            filepaths (List[str]): Paths to JSON graph files.
            epochs (int, optional): Number of training epochs. Default is 20.
            leaf_phase_epochs (int, optional): Number of epochs using leaf pruning. Default is 5.
            prune_step (float, optional): Incremental pruning ratio per epoch. Default is 0.05.
            prune_max_ratio (float, optional): Maximum pruning ratio. Default is 0.30.

        Returns:
            None
        """
        self.fit_encoders(filepaths)
        self.train()

        for epoch in range(epochs):
            ratio = min(prune_step * (epoch + 1), prune_max_ratio)
            dataset: List[torch_geometric.data.Data] = []
            pos_total = 0
            mask_total = 0

            for path in filepaths:
                G = self._load_graph_from_json(path)
                if epoch < leaf_phase_epochs:
                    G = self._delete_random_leaves(G, ratio=ratio)
                else:
                    G = self._delete_random_subtrees(G, ratio=ratio)

                data = self._graph_to_pyg(G)
                y = self._get_label_tensor(G)
                data.y = y

                # Apply mask: restrict to header-subtree nodes
                header_ids = self._header_subtree_mask(G)
                nid_list = getattr(data, "_node_ids", [])
                mask = torch.zeros(len(nid_list), dtype=torch.bool, device=self.device)
                for i, nid in enumerate(nid_list):
                    if nid in header_ids:
                        mask[i] = True
                if mask.sum().item() == 0:
                    continue

                data.train_mask = mask
                pos_total += int(y[mask].sum().item())
                mask_total += int(mask.sum().item())
                dataset.append(data)

            print(f"Epoch {epoch} | HeaderFieldClassifier: mask_nodes={mask_total} | pos_in_mask={pos_total}")
            self._train_epoch(dataset, epoch)

    def _train_epoch(self, dataset: List[torch_geometric.data.Data], epoch: int):
        """
        Train the model for a single epoch on the provided dataset.

        Loss is computed only on nodes belonging to header subtrees,
        using binary cross-entropy with `pos_weight` correction
        to handle class imbalance.

        Args:
            dataset (List[torch_geometric.data.Data]): List of graph samples.
            epoch (int): Current epoch index.
        """
        self.train()
        total_loss = 0.0

        for data in dataset:
            data = data.to(self.device)
            mask = getattr(data, "train_mask", None)
            if mask is None or mask.sum().item() == 0:
                continue

            x = self._encode_node_features(data._raw_node_attrs) if hasattr(data, "_raw_node_attrs") else data.x
            emb = self.gnn(x, data.edge_index)
            logits = self.head(emb).squeeze(-1)

            labels = data.y
            labels_m = labels[mask]
            logits_m = logits[mask]

            pos = labels_m.sum().item()
            neg = labels_m.numel() - pos
            if labels_m.numel() == 0:
                continue

            pos_weight = torch.tensor([neg / max(pos, 1)], device=self.device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits_m, labels_m, pos_weight=pos_weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())

        print(f"Epoch {epoch} | HeaderFieldClassifier loss: {total_loss:.4f}")

    def predict_subgraph(self, graph_path: str, node_embeddings: torch.Tensor) -> List[int]:
        """
        Predict which nodes in a graph correspond to struct fields
        within header declarations.

        Args:
            graph_path (str): Path to the original graph file (not directly used).
            node_embeddings (torch.Tensor): Node embeddings from the GNN.

        Returns:
            List[int]: Indices of nodes predicted as struct fields.
        """
        with torch.no_grad():
            logits = self.head(node_embeddings).squeeze(-1)
            scores = torch.sigmoid(logits)
        return [i for i, s in enumerate(scores.tolist()) if s > 0.5]

    def save_model(self, path: str):
        """
        Save the model, optimizer state, and encoders to disk.

        Files created:
            - `header_field_model.pt`: model and optimizer state.
            - `header_field_class_encoder.pkl`: class encoder.
            - `header_field_value_encoder.pkl`: value encoder.

        Args:
            path (str): Directory path where files are saved.
        """
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, os.path.join(path, "header_field_model.pt"))

        with open(os.path.join(path, "header_field_class_encoder.pkl"), "wb") as f:
            pickle.dump(self.class_encoder, f)
        with open(os.path.join(path, "header_field_value_encoder.pkl"), "wb") as f:
            pickle.dump(self.value_encoder, f)

        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load the model, optimizer state, and encoders from disk.

        Expected files:
            - `header_field_model.pt`
            - `header_field_class_encoder.pkl`
            - `header_field_value_encoder.pkl`

        Args:
            path (str): Directory path where the model is stored.
        """
        checkpoint = torch.load(os.path.join(path, "header_field_model.pt"), map_location=self.device)
        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        with open(os.path.join(path, "header_field_class_encoder.pkl"), "rb") as f:
            self.class_encoder = pickle.load(f)
        with open(os.path.join(path, "header_field_value_encoder.pkl"), "rb") as f:
            self.value_encoder = pickle.load(f)

        self.to(self.device)
        print(f"Model loaded from {path}")
