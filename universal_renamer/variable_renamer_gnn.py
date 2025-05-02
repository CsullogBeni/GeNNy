import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class VariableRenamerGNN:
    """
    Graph Neural Network-based variable renamer that learns from a single input graph.
    After training, it can rename all nodes with a given variable name to a new name.
    """

    def __init__(self, embedding_dim=64, epochs=20, lr=0.01):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr

        self.label_encoder = LabelEncoder()
        self.model = None
        self.optimizer = None

        self.node_id_map = {}
        self.reverse_node_id_map = {}
        self.graph_json = None

    def load_json_graph(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.graph_json = json.load(f)
        return self.graph_json

    def graph_to_data(self):
        G = nx.DiGraph()
        for node in self.graph_json["nodes"]:
            G.add_node(node["nodeId"], **node)
        for edge in self.graph_json["edges"]:
            G.add_edge(edge["source"], edge["target"])

        for idx, node_id in enumerate(G.nodes):
            self.node_id_map[node_id] = idx
            self.reverse_node_id_map[idx] = node_id

        features = []
        labels = []
        for node_id, attr in G.nodes(data=True):
            is_terminal = int(attr.get("class_") == "TerminalNodeImpl")
            has_value = int("value" in attr)
            features.append([is_terminal, has_value])
            labels.append(attr.get("value", ""))

        self.label_encoder.fit(labels)
        y = torch.tensor(self.label_encoder.transform(labels), dtype=torch.long)

        edge_index = torch.tensor([
            [self.node_id_map[src], self.node_id_map[tgt]]
            for src, tgt in G.edges
        ], dtype=torch.long).t().contiguous()

        x = torch.tensor(features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)

    class GCN(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    def train_model(self, data):
        self.model = self.GCN(input_dim=2, hidden_dim=self.embedding_dim, num_classes=len(self.label_encoder.classes_))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def rename_variable(self, old_name, new_name):
        renamed_graph = json.loads(json.dumps(self.graph_json))

        self.model.eval()
        data = self.graph_to_data()  # prepare PyG Data
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

        count = 0
        for idx, pred_idx in enumerate(pred):
            node_id = self.reverse_node_id_map[idx]
            predicted_value = self.label_encoder.inverse_transform([pred_idx.item()])[0]

            for node in renamed_graph["nodes"]:
                if node["nodeId"] == node_id and node.get("value") == old_name:
                    node["value"] = new_name
                    count += 1

        print(f"\nRenamed {count} predicted occurrences of '{old_name}' to '{new_name}'")
        return renamed_graph

    def compare_graphs(self, original, modified):
        print("Differences between original and modified graph:")
        for n1, n2 in zip(original["nodes"], modified["nodes"]):
            if n1.get("value") != n2.get("value"):
                print(f"Node {n1['nodeId']}: {n1.get('value')} -> {n2.get('value')}")

    def save(self, path="gnn_model.pt"):
        torch.save(self.model.state_dict(), path)
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

    def load(self, path="gnn_model.pt"):
        with open("label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        self.model = self.GCN(input_dim=2, hidden_dim=self.embedding_dim, num_classes=len(self.label_encoder.classes_))
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


if __name__ == "__main__":
    gnn = VariableRenamerGNN()

    folder = "data"
    '''graph_files = [f for f in os.listdir(folder) if f.endswith(".json")]

    print(f"\nFound {len(graph_files)} graph files in '{folder}'.")

    for file in graph_files:
        path = os.path.join(folder, file)
        print(f"\n=== Training on graph: {file} ===")
        gnn.load_json_graph(path)
        data = gnn.graph_to_data()
        gnn.train_model(data)

    gnn.save()'''
    gnn.load()

    # Test renaming after training
    print("\n=== Rename graph ===")
    test_path = os.path.join(folder, "basic_p4_v2_normalized.json")
    original_graph = gnn.load_json_graph(test_path)

    renamed_graph = gnn.rename_variable("egress_spec", "egress_specific")
    gnn.compare_graphs(original_graph, renamed_graph)

    renamed_graph = gnn.rename_variable("dstAddr", "destination_address")
    gnn.compare_graphs(original_graph, renamed_graph)
