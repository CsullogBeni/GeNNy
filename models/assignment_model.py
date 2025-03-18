import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder

from models.gnn import GNN
from pretty_printer.pretty_printer import PrettyPrinter


class Trainer:
    def __init__(self):
        self._classes = None
        self._values = None
        self._class_encoder = OneHotEncoder(handle_unknown='ignore')
        self._value_encoder = OneHotEncoder(handle_unknown='ignore')
        self._node_vectors = None
        self._edge_index = None
        self._node_features = None
        self._data = None
        self._model = None
        self._optimizer = None
        self._reconstructed_data = None

    def train(self, g: nx.DiGraph):
        self._classes = list(set(nx.get_node_attributes(g, 'class_').values()))
        self._values = list(set(filter(None, nx.get_node_attributes(g, 'value').values())))

        self._class_encoder.fit(np.array(self._classes).reshape(-1, 1))

        self._value_encoder.fit(np.array(self._values).reshape(-1, 1))
        self._node_vectors = {node: self._node_to_vector(g.nodes[node]) for node in g.nodes}

        edge_array = nx.to_numpy_array(g)
        edge_index = np.where(edge_array)
        self._edge_index = torch.tensor(edge_index, dtype=torch.long)

        self._node_features = np.array([self._node_to_vector(g.nodes[n]) for n in g.nodes], dtype=np.float64)
        self._node_features = torch.tensor(self._node_features, dtype=torch.float)

        self._data = Data(x=self._node_features, edge_index=self._edge_index)

        self._model = GNN(in_channels=self._node_features.shape[1], hidden_channels=16,
                          out_channels=self._node_features.shape[1])
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

        original_data = self._data
        number_of_training_epochs = 20000
        for epoch in range(number_of_training_epochs):
            if epoch % 100 == 0 and epoch > 0:
                self._data = self._modify_graph(original_data, removal_ratio=(epoch / number_of_training_epochs) * 0.5)
            loss = self._train()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

            if epoch == 200:
                self._data = self._modify_graph(self._data, removal_ratio=0.2)

    def _node_to_vector(self, attrs):
        node_id = attrs.get("nodeId", 0) / 10
        line = attrs.get("line", 0) / 10
        start = attrs.get("start", 0) / 10
        end = attrs.get("end", 0) / 10

        value = attrs.get("value", None)
        class_ = attrs.get("class_", "Unknown")

        class_encoded = self._class_encoder.transform([[class_]]).toarray()[0]

        if value is not None:
            value_encoded = self._value_encoder.transform([[value]]).toarray()[0]
        else:
            value_encoded = np.zeros(self._value_encoder.categories_[0].shape[0])

        return np.concatenate((np.array([node_id, line, start, end]), value_encoded, class_encoded))

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
        node_loss = nn.MSELoss()(output[:, :4] * 10, target[:, :4] * 100)  # Nagyobb súlyt adunk az első 4 értéknek
        feature_loss = nn.MSELoss()(output[:, 4:], target[:, 4:])  # Az egyéb jellemzők normál súlyon maradnak

        pred_edges = torch.mm(output, output.T)
        edge_target = torch.zeros_like(pred_edges)
        edge_target[edge_index[0], edge_index[1]] = 1
        edge_loss = nn.BCEWithLogitsLoss()(pred_edges, edge_target)

        return node_loss + feature_loss + edge_loss

    def generate(self, left_attr: str, right_attr: str, num_nodes=10):
        """Teljesen új gráfot generál a tanult mintázatok alapján, csak a modell és az encoderek használatával."""

        self._model.eval()
        with torch.no_grad():
            # Véletlenszerű bemenet generálása
            input_noise = torch.randn((num_nodes, self._model.encoder.in_channels))

            # Modell futtatása
            encoded = self._model.encoder(input_noise, torch.empty((2, 0), dtype=torch.long))
            generated_output = self._model.decoder(encoded, torch.empty((2, 0), dtype=torch.long))

        # Új gráf létrehozása
        new_graph = nx.DiGraph()

        # Oszlopok számának ellenőrzése
        class_encoding_size = len(self._class_encoder.categories_[0])  # Ennyi oszlopot vár az encoder
        value_encoding_size = len(self._value_encoder.categories_[0]) if self._value_encoder.categories_ else 0

        for i, node_data in enumerate(generated_output):
            # A megfelelő szeletelés biztosítása
            class_vector = node_data[-class_encoding_size:].reshape(1, -1)
            value_vector = node_data[4: 4 + value_encoding_size].reshape(1, -1) if value_encoding_size > 0 else None

            # Osztály dekódolása
            class_decoded = self._class_encoder.inverse_transform(class_vector)[0][0]

            # Érték dekódolása
            value_decoded = None
            if class_decoded == "TerminalNodeImpl":
                if value_vector is not None:
                    value_decoded = self._value_encoder.inverse_transform(value_vector)[0][0]
                value_decoded = left_attr if i == 0 else right_attr

            new_graph.add_node(i, class_=class_decoded, value=value_decoded)

        # **Élek generálása**
        edge_probabilities = torch.mm(generated_output, generated_output.T)
        edge_threshold = 0.5
        for i in range(num_nodes):
            for j in range(num_nodes):
                if edge_probabilities[i, j] > edge_threshold:
                    new_graph.add_edge(i, j)

        return new_graph


from graph.p4_graph import P4Graph

og_graph = P4Graph()
og_graph.read_from_json(
    r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\full_graphs\assignment.json")
assignment_model = Trainer()
assignment_model.train(og_graph.get_graph)
left_attr = "hdr.ethernet.dstAddr"
right_attr = "dstAddr"
reconstructed_graph = assignment_model.generate(left_attr, right_attr)
print(reconstructed_graph)

'''differences = og_graph.compare_graphs(reconstructed_graph)
print(differences['missing_edges'])
print(differences['extra_edges'])
for item in differences['node_attribute_differences']:
    print(f"Node {item['node']}:")
    print('Original:', item['original'])
    print('Reconstructed:', item['reconstructed'])
print(PrettyPrinter(og_graph.get_graph).get_script)
print(PrettyPrinter(reconstructed_graph).get_script)'''
