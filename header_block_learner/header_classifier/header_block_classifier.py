# header_block_classifier.py

import os
import pickle
from typing import List

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from genny.abs_genny import AbsGenny


class HeaderBlockClassifier(AbsGenny):
    """
    A graph-based neural classifier for detecting header block nodes in
    abstract syntax graphs.

    This model extends `AbsGenny` and is specialized for
    classifying nodes that correspond to `HeaderTypeDeclarationContext`.
    It uses Graph Neural Networks (GNNs) combined with a linear head to
    produce binary predictions at the node level.

    Key features:
        - Encodes AST nodes into vector representations.
        - Trains a GNN to propagate structural information.
        - Applies a binary classification head for detecting header block nodes.
        - Supports pruning strategies (removing leaves or subtrees) during
          training to improve generalization.

    Training process:
        1. Encoders are fitted on input JSON graphs.
        2. Graphs are loaded and augmented via pruning.
        3. Each graph is converted to a PyTorch Geometric `Data` object.
        4. Labels are generated to mark header declaration nodes as positive.
        5. The model is trained with a weighted binary cross-entropy loss
           to handle class imbalance.

    Saving and loading:
        - The model can be saved to disk, including its optimizer state and
          the encoders for classes and values.
        - A trained model can be restored for inference or further training.

    Args:
        hidden_dim (int, optional): Size of the hidden representation
            in the GNN. Default is 64.
        device (str, optional): Device to run the model on, e.g. `"cpu"`
            or `"cuda"`. Default is `"cpu"`.
    """

    def __init__(self, hidden_dim: int = 64, device: str = "cpu"):
        super().__init__(hidden_dim=hidden_dim, device=device)
        self.head = nn.Linear(self.hidden_dim, 1)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def _label_nodes(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        Generate binary labels for nodes in the input graph.

        A node is labeled as positive (1.0) if it matches the structural
        pattern leading to a `HeaderTypeDeclarationContext`:

            InputContext → DeclarationContext → TypeDeclarationContext
            → DerivedTypeDeclarationContext → HeaderTypeDeclarationContext

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
                                labels[id_to_index[c4]] = 1.0
        return labels

    def fit(self, filepaths: List[str], epochs: int = 20,
            leaf_phase_epochs: int = 5, prune_step: float = 0.05, prune_max_ratio: float = 0.30):
        """
        Train the classifier on a dataset of JSON graphs.

        The training is divided into two phases:
            - Early epochs use leaf pruning (removing leaves at random).
            - Later epochs use subtree pruning (removing larger structures).

        Args:
            filepaths (List[str]): List of paths to JSON graph files.
            epochs (int, optional): Number of training epochs. Default is 20.
            leaf_phase_epochs (int, optional): Number of initial epochs
                with leaf pruning. Default is 5.
            prune_step (float, optional): Incremental pruning ratio per epoch.
                Default is 0.05.
            prune_max_ratio (float, optional): Maximum pruning ratio. Default is 0.30.
        """
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
        """
        Train the model for a single epoch on the provided dataset.

        Uses binary cross-entropy loss with a `pos_weight` term to handle
        class imbalance between positive and negative samples.

        Args:
            dataset (List[torch_geometric.data.Data]): List of graph data
                samples, each containing node features, edges, and labels.
            epoch (int): Current epoch index, used for logging.

        Returns:
            None
        """
        self.train()
        total_loss = 0.0

        for data in dataset:
            data = data.to(self.device)
            x = self._encode_node_features(data._raw_node_attrs) if hasattr(data, "_raw_node_attrs") else data.x

            emb = self.gnn(x, data.edge_index)
            logits = self.head(emb).squeeze(-1)
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
        """
        Predict which nodes in a graph correspond to header declarations.

        Args:
            graph_path (str): Path to the original graph file (not used
                directly in prediction but kept for interface compatibility).
            node_embeddings (torch.Tensor): Node embeddings from the GNN.

        Returns:
            List[int]: Indices of nodes predicted as header block declarations.
        """
        with torch.no_grad():
            logits = self.head(node_embeddings).squeeze(-1)
            scores = torch.sigmoid(logits)
        return [i for i, s in enumerate(scores) if s > 0.5]

    def save_model(self, path: str):
        """
        Save the model, optimizer state, and encoders to disk.

        Files created:
            - `header_block_model.pt`: model and optimizer state dicts.
            - `header_block_class_encoder.pkl`: class encoder.
            - `header_block_value_encoder.pkl`: value encoder.

        Args:
            path (str): Directory path where the model should be saved.

        Returns:
            None
        """
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
        """
        Load the model, optimizer state, and encoders from disk.

        Expected files:
            - `header_block_model.pt`
            - `header_block_class_encoder.pkl`
            - `header_block_value_encoder.pkl`

        Args:
            path (str): Directory path where the model is stored.

        Returns:
            None
        """
        checkpoint = torch.load(os.path.join(path, "header_block_model.pt"), map_location=self.device)
        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        with open(os.path.join(path, "header_block_class_encoder.pkl"), "rb") as f:
            self.class_encoder = pickle.load(f)
        with open(os.path.join(path, "header_block_value_encoder.pkl"), "rb") as f:
            self.value_encoder = pickle.load(f)

        self.to(self.device)
        print(f"Model loaded from {path}")
