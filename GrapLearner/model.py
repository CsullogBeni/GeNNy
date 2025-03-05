import os
import pathlib
import torch
import networkx as nx

from torch_geometric.data import Data

from GrapLearner.p4_graph import P4Graph
from GrapLearner.trainer import Trainer


class Model:
    def __init__(self):
        self._graphs = []
        self._trainer = Trainer()
        self._gnn = self._trainer.get_model()

    def add_graph(self, g: P4Graph):
        self._graphs.append(g)

    def add_graphs(self, file_path: str):
        graph = P4Graph()
        if file_path.endswith(".json"):
            graph.read_from_json(file_path)
        elif file_path.endswith(".pkl"):
            graph.read_from_pkl(file_path)
        else:
            raise ValueError("Invalid file format")
        self._graphs.append(graph)

    def train(self):
        for g in self._graphs:
            self._trainer.train(g.get_graph())

    def reconstruct_graph(self, data: Data):
        self._gnn = self._trainer.get_model()
        if self._gnn is None:
            raise ValueError("Model not trained yet")
        self._gnn.eval()
        with torch.no_grad():
            reconstructed_x = self._gnn(data)
        return Data(x=reconstructed_x, edge_index=data.edge_index)

    def save(self, directory: str = pathlib.Path(__file__).parent.absolute(), filename: str = None):
        if not filename:
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            model_path = os.path.join(directory, date + "model.pth")
        else:
            model_path = os.path.join(directory, filename)

        torch.save(self._gnn.state_dict(), model_path)

    def load(self, model_path: str):
        if not (model_path.endswith(".pth") or not os.path.isfile(model_path)):
            raise ValueError("Invalid file format or file does not exist")
        self._trainer.load(model_path)
        self._gnn = self._trainer.get_model()

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
