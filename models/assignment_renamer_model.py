import json
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim


class AssignmentRenamer:
    """
    A class that processes assignment data stored as graphs and applies machine learning
    to rename specific node values based on learned patterns.

    It builds a directed graph from JSON data, trains a neural network to identify
    renamable nodes, and applies the learned model to update the graph accordingly.
    """

    def __init__(self, original_path, renamed_path, embedding_dim=16, lr=0.01, epochs=100):
        """
        Initializes the AssignmentRenamer with file paths and model parameters.

        :param original_path: Path to the original assignment JSON file.
        :param renamed_path: Path to the renamed assignment JSON file.
        :param embedding_dim: Dimensionality of node embeddings.
        :param lr: Learning rate for model training.
        :param epochs: Number of training epochs.
        """
        self.original_data = self.load_json(original_path)
        self.renamed_data = self.load_json(renamed_path)
        self.graph_original = self.build_graph(self.original_data)
        self.graph_renamed = self.build_graph(self.renamed_data)
        self.embedding_dim = embedding_dim
        self.epochs = epochs

        self.node_embedding = self.NodeEmbedding(len(self.graph_original.nodes), embedding_dim)
        self.model = self.NameChangeModel(embedding_dim)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.node_embedding.parameters()), lr=lr
        )
        self.loss_fn = nn.BCELoss()

    class NodeEmbedding(nn.Module):
        """
        A neural network module that creates embeddings for graph nodes.
        """

        def __init__(self, num_nodes, embedding_dim=16):
            super().__init__()
            self.embedding = nn.Embedding(num_nodes, embedding_dim)

        def forward(self, node_ids):
            """
            Generates embedding vectors for given node IDs.

            :param node_ids: A tensor containing node indices.
            :return: A tensor of embeddings.
            """
            return self.embedding(node_ids)

    class NameChangeModel(nn.Module):
        """
        A simple feed-forward neural network for predicting name changes in nodes.
        """

        def __init__(self, embedding_dim=16):
            super().__init__()
            self.fc1 = nn.Linear(embedding_dim, 32)
            self.fc2 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            """
            Passes input through the network and predicts name change probability.

            :param x: Input feature vector.
            :return: A probability value indicating name change likelihood.
            """
            x = self.relu(self.fc1(x))
            return self.sigmoid(self.fc2(x))

    def load_json(self, file_path):
        """
        Loads a JSON file into a Python dictionary.

        :param file_path: Path to the JSON file.
        :return: Parsed JSON data as a dictionary.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_graph(self, data):
        """
        Constructs a directed graph from JSON data.

        :param data: Dictionary containing node and edge data.
        :return: A NetworkX directed graph.
        """
        G = nx.DiGraph()
        for node in data["nodes"]:
            attributes = {k: v for k, v in node.items() if k != "nodeId"}
            if node["class_"] == "TerminalNodeImpl":
                attributes["value"] = node["value"]
            else:
                attributes.pop("value", None)
            G.add_node(node["nodeId"], **attributes)
        for edge in data["edges"]:
            G.add_edge(edge["source"], edge["target"])
        return G

    def extract_features(self, graph):
        """
        Extracts node features and target labels for training.

        :param graph: A NetworkX graph.
        :return: Node indices and target labels.
        """
        node_ids, targets = [], []
        for node, data in graph.nodes(data=True):
            if data.get("class_") == "TerminalNodeImpl":
                node_ids.append(node)
                target_value = 1.0 if data.get("value") == "destinationAddress" else 0.0
                targets.append(torch.tensor([target_value]))
        return torch.tensor(node_ids, dtype=torch.long), torch.stack(targets)

    def train_model(self):
        """
        Trains the neural network model to recognize name changes.
        """
        input_nodes, target_labels = self.extract_features(self.graph_original)
        _, renamed_labels = self.extract_features(self.graph_renamed)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            embeddings = self.node_embedding(input_nodes)
            output = self.model(embeddings)
            loss = self.loss_fn(output, renamed_labels)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    def rename_graph(self, rename_to="destinationAddress"):
        """
        Uses the trained model to rename nodes in the graph based on learned patterns.

        :param rename_to: The new value to assign to renamed nodes.
        :return: A new graph with renamed nodes.
        """
        new_graph = self.graph_original.copy()
        for node, data in new_graph.nodes(data=True):
            if data.get("class_") == "TerminalNodeImpl":
                input_vector = self.node_embedding(torch.tensor([node], dtype=torch.long)).squeeze(0)
                prediction = self.model(input_vector).item()
                if prediction > 0.75:
                    data["value"] = rename_to
        return new_graph

    def process(self, rename_to="destinationAddress"):
        """
        Executes the complete process: training, renaming, and returning updated data.

        :param rename_to: The new value to assign to renamed nodes.
        :return: A dictionary representing the renamed graph.
        """
        self.train_model()
        renamed_graph = self.rename_graph(rename_to)
        renamed_data = {"nodes": [], "edges": []}
        for node in renamed_graph.nodes:
            current = renamed_graph.nodes[node]
            current["nodeId"] = node
            renamed_data["nodes"].append(current)
        for edge in renamed_graph.edges:
            renamed_data["edges"].append({"source": edge[0], "target": edge[1]})
        print("Processing completed.")
        return renamed_data


# Instantiate and run the process
if __name__ == "__main__":
    renamer = AssignmentRenamer("assignment.json", "assignment_renamed.json")
    with open("assignment.json", "r", encoding="utf-8") as f:
        original_data = json.load(f)
    renamed_output = renamer.process()
    print("Differences:")
    for index in range(len(original_data['nodes'])):
        try:
            difference = 'Node ' + str(index) + ': '
            current_original = original_data['nodes'][index]
            current_renamed = renamed_output['nodes'][index]
            if current_original['nodeId'] != current_renamed['nodeId']:
                difference += 'nodeId: ' + str(current_original['nodeId']) + ' -> ' + str(
                    current_renamed['nodeId']) + ', '
            if current_original['line'] != current_renamed['line']:
                difference += 'line: ' + str(current_original['line']) + ' -> ' + str(current_renamed['line']) + ', '
            if current_original['start'] != current_renamed['start']:
                difference += 'start: ' + str(current_original['start']) + ' -> ' + str(current_renamed['start']) + ', '
            if current_original['end'] != current_renamed['end']:
                difference += 'end: ' + str(current_original['end']) + ' -> ' + str(current_renamed['end']) + ', '
            if current_original['class_'] != current_renamed['class_']:
                difference += 'class_: ' + str(current_original['class_']) + ' -> ' + str(
                    current_renamed['class_']) + ', '
            try:
                if current_original['value'] != current_renamed['value']:
                    difference += 'value: ' + str(current_original['value']) + ' -> ' + str(
                        current_renamed['value']) + ', '
            except KeyError:
                pass
            if difference != 'Node ' + str(index) + ': ':
                print(difference)
        except Exception as e:
            print(e)
            continue
