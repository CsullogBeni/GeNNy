import os
import pickle
from typing import List

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from graph_learner.abstract_graph_learner import AbstractGraphLearner


class HeaderFieldClassifier(AbstractGraphLearner):
    def __init__(self, hidden_dim: int = 64, device: str = "cpu",
                 gnn_layers: int = 5, gnn_dropout: float = 0.10):
        # több réteg + kis dropout a multi-hop mintázatokhoz
        super().__init__(hidden_dim=hidden_dim, device=device, gnn_layers=gnn_layers, gnn_dropout=gnn_dropout)
        self.head = nn.Linear(self.hidden_dim, 1)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    # --- HELYES CÍMKÉZÉS: header alatti StructFieldContext = 1 ---
    def _label_nodes(self, graph: nx.DiGraph) -> torch.Tensor:
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
                            # BFS a header-subtree-n
                            q, vis = [c4], set()
                            while q:
                                cur = q.pop(0)
                                if cur in vis: continue
                                vis.add(cur)
                                if graph.nodes[cur].get("class_") == "StructFieldContext":
                                    labels[id_to_index[cur]] = 1.0
                                q.extend(list(graph.successors(cur)))
        return labels

    # --- Maszk: csak a header-subtree csomópontok (negatívok is itt!) ---
    @staticmethod
    def _header_subtree_mask(graph: nx.DiGraph) -> set:
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
                            # jelöljük a teljes header-subtree-t
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

                # ➕ maszk: csak a header-subtree csomópontok tanuljanak
                header_ids = self._header_subtree_mask(G)
                # a node-sorrendet az _graph_to_pyg tette data._node_ids-be
                nid_list = getattr(data, "_node_ids", [])
                mask = torch.zeros(len(nid_list), dtype=torch.bool, device=self.device)
                for i, nid in enumerate(nid_list):
                    if nid in header_ids:
                        mask[i] = True
                # ha a pruning kinyírta a header-subtree-t → ugorjuk a mintát
                if mask.sum().item() == 0:
                    continue

                data.train_mask = mask
                pos_total += int(y[mask].sum().item())
                mask_total += int(mask.sum().item())
                dataset.append(data)

            print(f"Epoch {epoch} | HeaderFieldClassifier: mask_nodes={mask_total} | pos_in_mask={pos_total}")
            self._train_epoch(dataset, epoch)

    def _train_epoch(self, dataset: List[torch_geometric.data.Data], epoch: int):
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
        with torch.no_grad():
            logits = self.head(node_embeddings).squeeze(-1)
            scores = torch.sigmoid(logits)
        return [i for i, s in enumerate(scores.tolist()) if s > 0.5]

    def save_model(self, path: str):
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
        checkpoint = torch.load(os.path.join(path, "header_field_model.pt"), map_location=self.device)
        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        with open(os.path.join(path, "header_field_class_encoder.pkl"), "rb") as f:
            self.class_encoder = pickle.load(f)
        with open(os.path.join(path, "header_field_value_encoder.pkl"), "rb") as f:
            self.value_encoder = pickle.load(f)

        self.to(self.device)
        print(f"Model loaded from {path}")
