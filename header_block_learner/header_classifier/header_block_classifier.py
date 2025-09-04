# header_block_classifier.py

import os
import pickle
from typing import List

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from graph_learner.abstract_graph_learner import AbstractGraphLearner


class HeaderBlockClassifier(AbstractGraphLearner):
    def __init__(self, hidden_dim: int = 64, device: str = "cpu"):
        super().__init__(hidden_dim=hidden_dim, device=device)
        self.head = nn.Linear(self.hidden_dim, 1)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    # --- HELYES CÍMKÉZÉS: HeaderTypeDeclarationContext = 1 ---
    def _label_nodes(self, graph: nx.DiGraph) -> torch.Tensor:
        labels = torch.zeros(len(graph.nodes), dtype=torch.float32, device=self.device)
        id_to_index = {nid: i for i, nid in enumerate(graph.nodes)}

        for node_id in graph.nodes:
            # keresünk InputContext → DeclarationContext → TypeDeclarationContext → DerivedTypeDeclarationContext → HeaderTypeDeclarationContext
            if graph.nodes[node_id].get("class_") != "InputContext":
                continue
            for c1 in graph.successors(node_id):
                if graph.nodes[c1].get("class_") != "DeclarationContext":
                    continue
                for c2 in graph.successors(c1):
                    if graph.nodes[c2].get("class_") != "TypeDeclarationContext":
                        continue
                    for c3 in graph.successors(c2):
                        if graph.nodes[c3].get("class_") != "DerivedTypeDeclarationContext":
                            continue
                        for c4 in graph.successors(c3):
                            if graph.nodes[c4].get("class_") == "HeaderTypeDeclarationContext":
                                # CSAK a header deklarációs csomópont legyen 1
                                labels[id_to_index[c4]] = 1.0
        return labels

    def fit(self, filepaths: List[str], epochs: int = 20,
            leaf_phase_epochs: int = 5, prune_step: float = 0.05, prune_max_ratio: float = 0.30):
        self.fit_encoders(filepaths)
        self.train()

        for epoch in range(epochs):
            ratio = min(prune_step * (epoch + 1), prune_max_ratio)
            dataset = []

            for path in filepaths:
                G = self._load_graph_from_json(path)
                if epoch < leaf_phase_epochs:
                    G = self._delete_random_leaves(G, ratio=ratio)
                else:
                    G = self._delete_random_subtrees(G, ratio=ratio)

                data = self._graph_to_pyg(G)
                y = self._get_label_tensor(G)  # [N]
                data.y = y
                dataset.append(data)

            self._train_epoch(dataset, epoch)

    def _train_epoch(self, dataset: List[torch_geometric.data.Data], epoch: int):
        self.train()
        total_loss = 0.0

        for data in dataset:
            data = data.to(self.device)
            x = self._encode_node_features(data._raw_node_attrs) if hasattr(data, "_raw_node_attrs") else data.x

            emb = self.gnn(x, data.edge_index)
            logits = self.head(emb).squeeze(-1)
            # pos_weight az osztály-imbalance ellen
            pos = data.y.sum().clamp(min=1.0)
            neg = (data.y.numel() - pos).clamp(min=1.0)
            pos_weight = (neg / pos)
            loss = F.binary_cross_entropy_with_logits(logits, data.y, pos_weight=pos_weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())

        print(f"Epoch {epoch} | HeaderBlockClassifier loss: {total_loss:.4f}")

    def predict_subgraph(self, graph_path: str, node_embeddings: torch.Tensor) -> List[int]:
        with torch.no_grad():
            logits = self.head(node_embeddings).squeeze(-1)
            scores = torch.sigmoid(logits)
        return [i for i, s in enumerate(scores) if s > 0.5]

    def save_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, os.path.join(path, "header_block_model.pt"))

        with open(os.path.join(path, "header_block_class_encoder.pkl"), "wb") as f:
            pickle.dump(self.class_encoder, f)
        with open(os.path.join(path, "header_block_value_encoder.pkl"), "wb") as f:
            pickle.dump(self.value_encoder, f)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(os.path.join(path, "header_block_model.pt"), map_location=self.device)
        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        with open(os.path.join(path, "header_block_class_encoder.pkl"), "rb") as f:
            self.class_encoder = pickle.load(f)
        with open(os.path.join(path, "header_block_value_encoder.pkl"), "rb") as f:
            self.value_encoder = pickle.load(f)

        self.to(self.device)
        print(f"Model loaded from {path}")
