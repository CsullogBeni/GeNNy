import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from GrapLearner.p4_graph import P4Graph


class Trainer:
    def __init__(self, g: nx.DiGraph):
        self._classes = list(set(nx.get_node_attributes(g, 'class_').values()))
        self._encoder = OneHotEncoder(handle_unknown='ignore')
        self._encoder.fit(np.array(self._classes).reshape(-1, 1))
        self._node_vectors = {node: self._node_to_vector(g.nodes[node]) for node in g.nodes}

        '''self._adj_matrix = nx.to_numpy_array(g)'''
        edge_array = nx.to_numpy_array(g)
        edge_index = np.where(edge_array)
        self._edge_index = torch.tensor(edge_index, dtype=torch.long)

        self._node_features = np.array([self._node_to_vector(g.nodes[n]) for n in g.nodes], dtype=np.float64)
        self._node_features = torch.tensor(self._node_features, dtype=torch.float)

        self._data = Data(x=self._node_features, edge_index=self._edge_index)

        self._model = GNN(in_channels=self._node_features.shape[1], hidden_channels=16,
                          out_channels=self._node_features.shape[1])
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.01)
        # self._loss_fn = nn.MSELoss()

        original_data = self._data
        for epoch in range(20000):
            if epoch % 100 == 0 and epoch > 0:
                self._data = self._modify_graph(original_data, removal_ratio=(epoch / 10000) * 0.5)
            loss = self._train()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

            if epoch == 200:
                self._data = self._modify_graph(self._data, removal_ratio=0.2)
        '''self.load()'''
        self._reconstructed_data = self._reconstruct_graph(original_data)
        '''self.save()'''

        '''assert original_data.x.shape == self._reconstructed_data.x.shape
        assert original_data.edge_index.shape == self._reconstructed_data.edge_index.shape'''

        '''fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self.draw_graph(axes[0], original_data, "Original Graph", with_arrows=False)
        self.draw_graph(axes[1], self._data, "Incomplete Graph", with_arrows=False)
        self.draw_graph(axes[2], self._reconstructed_data, "Reconstructed Graph", with_arrows=True)
        plt.show()'''

    @property
    def get_reconstructed_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_edges_from(self._reconstructed_data.edge_index.T.tolist())

        # Eredeti attribútumok visszaállítása
        node_attrs = {}
        for node in g.nodes:
            vector = self._reconstructed_data.x[node].numpy().tolist()
            # Eredeti attribútumok dekódolása
            node_attrs[node] = {
                "nodeId": int(vector[0]),
                "line": int(vector[1]),
                "start": int(vector[2]),
                "end": int(vector[3]),
                "value": vector[4] if vector[4] != -1 else None,
                "class_": self._encoder.categories_[0][np.argmax(vector[5:])].item()
            }
        nx.set_node_attributes(g, node_attrs)
        return g

    def _node_to_vector(self, attrs):
        node_id = attrs.get("nodeId", 0)
        line = attrs.get("line", 0)
        start = attrs.get("start", 0)
        end = attrs.get("end", 0)
        value = attrs.get("value", "None")
        class_ = attrs.get("class_", "Unknown")

        class_encoded = self._encoder.transform([[class_]])[0].toarray()[0]
        value_numeric = float(value) if value and value.replace('.', '', 1).isdigit() else -1
        return np.concatenate((np.array([node_id, line, start, end, value_numeric]), np.array(class_encoded)))

    def _train(self):
        self._model.train()
        self._optimizer.zero_grad()
        output = self._model(self._data)
        loss = self.loss_fn(output, self._data.x, self._data.edge_index)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    @staticmethod
    def _modify_graph(data, removal_ratio=0.2):
        num_edges = data.edge_index.shape[1]
        num_remove = int(removal_ratio * num_edges)
        mask = torch.ones(num_edges, dtype=torch.bool)

        # Legnagyobb indexű node-hoz kapcsolódó élek eltávolítása
        nodes = data.edge_index[0].unique()
        num_nodes = nodes.shape[0]

        # Ha túl nagy a num_remove, csökkentsük le
        num_remove = min(num_remove, num_nodes)

        if num_remove > 0:
            nodes_to_remove = nodes.topk(num_remove).values
            remove_indices = [i for i, (src, _) in enumerate(data.edge_index.T.tolist())
                              if src in nodes_to_remove]
            mask[remove_indices] = False

        new_edge_index = data.edge_index[:, mask]
        return Data(x=data.x, edge_index=new_edge_index)

    @staticmethod
    def loss_fn(output, target, edge_index):
        node_loss = nn.MSELoss()(output, target)
        pred_edges = torch.mm(output, output.T)
        edge_target = torch.zeros_like(pred_edges)
        edge_target[edge_index[0], edge_index[1]] = 1
        edge_loss = nn.BCEWithLogitsLoss()(pred_edges, edge_target)
        return node_loss + edge_loss

    @staticmethod
    def visualize_graph(original_data, modified_data, reconstructed_data):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Helper function for drawing graphs
        def draw_graph(ax, data, title):
            G = nx.DiGraph()
            G.add_edges_from(data.edge_index.T.tolist())
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)
            ax.set_title(title)

        draw_graph(axes[0], original_data, "Original Graph")
        draw_graph(axes[1], modified_data, "Incomplete Graph")
        draw_graph(axes[2], reconstructed_data, "Reconstructed Graph")

        plt.show()

    def _reconstruct_graph(self, data):
        self._model.eval()
        with torch.no_grad():
            reconstructed_x = self._model(data)
        return Data(x=reconstructed_x, edge_index=data.edge_index)

    @staticmethod
    def draw_graph(ax, data, title, with_arrows=True):
        G = nx.DiGraph()
        G.add_edges_from(data.edge_index.T.tolist())
        pos = nx.spring_layout(G)
        if with_arrows:
            nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', connectionstyle='arc3,rad=0.1')
        else:
            nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        ax.set_title(title)

    @staticmethod
    def compare_graphs(original_g: nx.DiGraph, reconstructed_g: nx.DiGraph):
        differences = {
            "missing_edges": [],
            "extra_edges": [],
            "node_attribute_differences": []
        }

        original_edges_to_convert = set(original_g.edges())
        reconstructed_edges = set(reconstructed_g.edges())

        original_edges = []
        for tuple in original_edges_to_convert:
            original_edges.append((int(tuple[0]), int(tuple[1])))
        '''print(sorted(original_edges))
        print(sorted(reconstructed_edges))'''

        differences["missing_edges"] = []
        for edge in original_edges:
            if edge not in reconstructed_edges:
                differences["missing_edges"].append(edge)

        differences["extra_edges"] = []
        for edge in reconstructed_edges:
            if edge not in original_edges:
                differences["extra_edges"].append(edge)

        for index in range(len(original_g.nodes)):
            original_node = original_g.nodes[str(index)].items()
            reconstructed_node = reconstructed_g.nodes[index].items()

            original_node = sorted(original_node)
            reconstructed_node = sorted(reconstructed_node)

            if original_node != reconstructed_node:
                differences["node_attribute_differences"].append({
                    "node": index,
                    "original": original_node,
                    "reconstructed": reconstructed_node
                })

        return differences

    def save(self):
        torch.save(self._model.state_dict(), "model.pth")

    def load(self):
        self._model = GNN(in_channels=self._node_features.shape[1], hidden_channels=16,
                          out_channels=self._node_features.shape[1])
        self._model.load_state_dict(torch.load("model.pth"))


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.encoder = GCNConv(in_channels, hidden_channels)
        self.decoder = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index).relu()
        x = self.dropout(x)
        x = self.decoder(x, edge_index)
        return x


graph = P4Graph()
graph.read_from_json(
    r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\full_graphs\assignment.json")
trainer = Trainer(graph.get_graph)
reconstructed_graph = trainer.get_reconstructed_graph

differences = Trainer.compare_graphs(graph.get_graph, reconstructed_graph)
print(differences['missing_edges'])
print(differences['extra_edges'])
for item in differences['node_attribute_differences']:
    print(f"Node {item['node']}:")
    print('Original:', item['original'])
    print('Reconstructed:', item['reconstructed'])
