import os
import json


class GraphNormalizer:
    """
    A class for normalizing the IR JSON file. That contains a P4 Abstract Syntax Tree (AST).
    ASTs are represented as a directed graph with nodes and edges. Graph normalization ensures
    that the graph is well-formed, filters out unnecessary nodes attributes. While connecting
    nodes, the edge contains source and target nodes' ids.

    Attributes:
        _file_path (str): The path to the IR JSON file.
        _data (dict): The normalized IR JSON data.
        _normalized (bool): A flag indicating whether the IR JSON file has been normalized.
    """

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._data = {}
        self._normalized = False

    def normalize(self) -> None:
        """
        Normalizes the IR JSON file by reading data, checking nodes and edges,
        and setting the normalized flag to True.
        """
        self._read_ir_json()
        self._check_nodes()
        self._check_edges()
        self._normalized = True

    def _read_ir_json(self) -> None:
        """
        Reads the given file path and loads its data into the _data attribute.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a file or is not a JSON file.
        """
        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"File {self._file_path} not found")
        if not os.path.isfile(self._file_path):
            raise ValueError(f"Path {self._file_path} is not a file")
        if not self._file_path.endswith('.json'):
            raise ValueError(f"File {self._file_path} is not a JSON file")
        with open(self._file_path, 'r') as f:
            self._data = json.load(f)

    def _check_nodes(self) -> None:
        """
        Checks the nodes in the IR JSON data and ensures they have the necessary properties.

        Raises:
            ValueError: If a node is missing any of the necessary properties.
        """
        for node in self._data['nodes']:
            if not all(key in node for key in ['start', 'end', 'nodeId', 'class_']):
                raise ValueError(f"Node {node['nodeId']} does not have all necessary properties")
            node['nodeId'] = int(node['nodeId'])
            node['start'] = int(node['start'])
            node['end'] = int(node['end'])
            node['line'] = int(node['line'])
            node['class_'] = str(node['class_'])
            if 'value' in node:
                node['value'] = str(node['value'])

    def _check_edges(self) -> None:
        """
        Checks the edges in the IR JSON data and ensures they have the necessary properties.

        Raises:
            ValueError: If an edge is missing any of the necessary properties.
        """
        edges = []
        for edge in self._data['edges']:
            source = self._find_node_id(int(edge['IN']['id']))
            target = self._find_node_id(int(edge['OUT']['id']))
            result = {"source": source, "target": target}
            edges.append(result)
        self._data['edges'] = edges

    def _find_node_id(self, id: int) -> int:
        """
        Finds the node id in the IR JSON data.

        Args:
             id (int): The id of the node to find.

        Returns:
            int: The node id.

        Raises:
            Exception: If the node id is not found.
        """
        for node in self._data['nodes']:
            if node['id'] == id:
                return int(node['nodeId'])
        raise Exception(f"Node id {id} not found")

    def export_normalized_graph(self, file_path: str) -> None:
        """
        Exports the normalized IR JSON data to a file.

        Args:
            file_path (str): The path to the file to export the data to.
        """
        if not self._normalized:
            raise Exception("Graph not normalized")
        with open(file_path, 'w') as f:
            json.dump(self._data, f)
        print(f"Normalized graph exported to {file_path}")
