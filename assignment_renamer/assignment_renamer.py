import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from sklearn.preprocessing import LabelEncoder


class UniversalAssignmentRenamer:
    """
    A universal graph-based assignment renamer that learns how to rename terminal nodes
    based on their structural context using embedding and classification.
    """
    def __init__(self, embedding_dim=32, lr=0.01, epochs=50):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.epochs = epochs

        self.graph_pairs = []
        self.node_graph_map = []
        self.label_targets = []
        self.label_encoder = LabelEncoder()

        self.node_embedding = None
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()

    class NodeEmbedding(nn.Module):
        """
        A learnable embedding layer for nodes, indexed by node training ID.
        """
        def __init__(self, num_nodes, embedding_dim=32):
            super().__init__()
            self.embedding = nn.Embedding(num_nodes, embedding_dim)

        def forward(self, node_ids):
            return self.embedding(node_ids)

    class NamePredictionModel(nn.Module):
        """
        A simple feed-forward neural network that predicts new names from node embeddings.
        """
        def __init__(self, embedding_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(embedding_dim, 64)
            self.fc2 = nn.Linear(64, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    def load_json(self, path):
        """
        Loads a JSON file and returns the parsed content.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_graph(self, data):
        """
        Constructs a directed graph from a node-edge dictionary structure.
        """
        G = nx.DiGraph()
        for node in data["nodes"]:
            G.add_node(node["nodeId"], **{k: v for k, v in node.items() if k != "nodeId"})
        for edge in data["edges"]:
            G.add_edge(edge["source"], edge["target"])
        return G

    def add_training_pair(self, original_path, renamed_path):
        """
        Adds a training graph pair, and identifies renamed terminal nodes for training.
        Only nodes whose values have changed are considered training samples.
        """
        G_orig = self.build_graph(self.load_json(original_path))
        G_renamed = self.build_graph(self.load_json(renamed_path))
        graph_idx = len(self.graph_pairs)
        self.graph_pairs.append((G_orig, G_renamed))

        for node_id, data in G_orig.nodes(data=True):
            if data.get("class_") == "TerminalNodeImpl":
                renamed_value = G_renamed.nodes[node_id].get("value")
                original_value = G_orig.nodes[node_id].get("value")
                if renamed_value != original_value:
                    self.node_graph_map.append((node_id, graph_idx))
                    self.label_targets.append(renamed_value)

    def finalize_training_data(self):
        """
        Prepares encoded labels and initializes the model and optimizer.
        """
        self.label_encoder.fit(self.label_targets)
        self.label_targets = [self.label_encoder.transform([val])[0] for val in self.label_targets]

        self.node_embedding = self.NodeEmbedding(len(self.node_graph_map), self.embedding_dim)
        self.model = self.NamePredictionModel(self.embedding_dim, len(self.label_encoder.classes_))
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.node_embedding.parameters()), lr=self.lr
        )

    def train(self):
        """
        Trains the model using node embeddings and cross-entropy loss on name labels.
        """
        node_indices = torch.tensor(range(len(self.node_graph_map)), dtype=torch.long)
        labels = torch.tensor(self.label_targets, dtype=torch.long)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            embeddings = self.node_embedding(node_indices)
            outputs = self.model(embeddings)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def save_model(self, path="model.pt", label_path="label_encoder.pkl"):
        """
        Saves the model and label encoder to disk.
        """
        torch.save({
            "model": self.model.state_dict(),
            "embedding": self.node_embedding.state_dict()
        }, path)
        with open(label_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

    def load_model(self, path="model.pt", label_path="label_encoder.pkl"):
        """
        Loads a previously saved model and label encoder from disk.
        """
        with open(label_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.node_embedding = self.NodeEmbedding(len(self.node_graph_map), self.embedding_dim)
        self.model = self.NamePredictionModel(self.embedding_dim, len(self.label_encoder.classes_))

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.node_embedding.load_state_dict(checkpoint["embedding"])

        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.node_embedding.parameters()), lr=self.lr
        )

    def get_context_signature(self, graph, node_id, depth=3):
        """
        Retrieves the context signature of a node by walking up its predecessors up to a given depth.
        The context consists of the class_ labels of ancestor nodes.
        """
        context = []
        current = node_id
        for _ in range(depth):
            preds = list(graph.predecessors(current))
            if not preds:
                break
            current = preds[0]
            context.append(graph.nodes[current].get("class_"))
        return tuple(context)

    def find_closest_node_index(self, node_data, input_graph, node_id):
        """
        Finds the most similar training node to the input node based on value, class, and context signature.
        Returns the training index or None if no match is found.
        """
        input_context = self.get_context_signature(input_graph, node_id)

        for idx, (train_node_id, graph_idx) in enumerate(self.node_graph_map):
            train_graph = self.graph_pairs[graph_idx][0]
            train_node = train_graph.nodes[train_node_id]

            if node_data.get("class_") != train_node.get("class_"):
                continue
            if node_data.get("value") != train_node.get("value"):
                continue

            train_context = self.get_context_signature(train_graph, train_node_id)
            if input_context != train_context:
                continue

            return idx
        return None

    def predict_and_rename(self, input_json, rename_to="destinationAddress", threshold=0.75):
        """
        Predicts new names for terminal nodes based on learned patterns.
        Only renames a node if the confidence for the target name exceeds the threshold.
        """
        input_graph = self.build_graph(input_json)
        renamed_data = {"nodes": [], "edges": input_json["edges"]}

        for node_id, data in input_graph.nodes(data=True):
            if data.get("class_") == "TerminalNodeImpl":
                idx = self.find_closest_node_index(data, input_graph, node_id)
                if idx is None:
                    continue

                with torch.no_grad():
                    emb = self.node_embedding(torch.tensor([idx]))
                    out = self.model(emb).squeeze(0)
                    probs = torch.softmax(out, dim=0)

                    '''print(f"Node {node_id}:")
                    print(f"  Original value: {data.get('value')}")
                    print(f"  Predicted top-1: {self.label_encoder.inverse_transform([probs.argmax().item()])[0]}")'''

                    if rename_to in self.label_encoder.classes_:
                        rename_idx = self.label_encoder.transform([rename_to])[0]
                        print(f"  {rename_to} prob: {probs[rename_idx].item():.4f}")
                        rename_prob = probs[rename_idx].item()

                        if rename_prob > threshold:
                            data["value"] = rename_to
                    else:
                        print(f"  {rename_to} not found in label encoder.")
                        continue

        for node_id, attrs in input_graph.nodes(data=True):
            new_node = attrs.copy()
            new_node["nodeId"] = node_id
            renamed_data["nodes"].append(new_node)

        return renamed_data

    def compare_results(self, original_jsons, renamed_graphs):
        """
        Compares the original and renamed graphs and prints all value changes for terminal nodes.
        """
        for idx, (orig, renamed) in enumerate(zip(original_jsons, renamed_graphs)):
            print(f"\nDifferences for input {idx}:")
            for i, (n1, n2) in enumerate(zip(orig["nodes"], renamed["nodes"])):
                changes = []
                for key in ["value"]:
                    if n1.get(key) != n2.get(key):
                        changes.append(f"{key}: {n1.get(key)} -> {n2.get(key)}")
                if changes:
                    print(f"Node {n1['nodeId']}: {', '.join(changes)}")


if __name__ == "__main__":
    renamer = UniversalAssignmentRenamer(epochs=25)

    file_pairs = [
        ("data/dst_assignment.json", "data/dst_assignment_renamed.json"),
        ("data/src_assignment.json", "data/src_assignment_renamed.json"),
        ("data/spec_assignment.json", "data/spec_assignment_renamed.json"),
    ]

    for orig, renamed in file_pairs:
        renamer.add_training_pair(orig, renamed)

    renamer.finalize_training_data()
    # renamer.train()
    # renamer.save_model()
    renamer.load_model()

    original_json = renamer.load_json("data/dst_assignment.json")
    renamed_graphs = renamer.predict_and_rename(original_json, rename_to="destinationAddress", threshold=0.9)
    renamer.compare_results([original_json], [renamed_graphs])

    original_json = renamer.load_json("data/src_assignment.json")
    renamed_graphs = renamer.predict_and_rename(original_json, rename_to="sourceAddress", threshold=0.5)
    renamer.compare_results([original_json], [renamed_graphs])

    original_json = renamer.load_json("data/spec_assignment.json")
    renamed_graphs = renamer.predict_and_rename(original_json, rename_to="egress_specific", threshold=0.5)
    renamer.compare_results([original_json], [renamed_graphs])
