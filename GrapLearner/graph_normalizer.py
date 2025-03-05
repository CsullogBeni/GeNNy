import os
import json


class GraphNormalizer:
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._data = {}
        self._normalized = False

    def normalize(self):
        self._read_ir_json()
        self._check_nodes()
        self._check_edges()
        self._normalized = True

    def _read_ir_json(self):
        with open(self._file_path, 'r') as f:
            self._data = json.load(f)

    def _check_nodes(self):
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

    def _check_edges(self):
        edges = []
        for edge in self._data['edges']:
            try:
                source = self._find_node_id(int(edge['IN']['id']))
                target = self._find_node_id(int(edge['OUT']['id']))
                result = {"source": source, "target": target}
                edges.append(result)
            except Exception as e:
                print(e)
        self._data['edges'] = edges

    def _find_node_id(self, id: int) -> int:
        for node in self._data['nodes']:
            if node['id'] == id:
                return int(node['nodeId'])
        raise Exception

    def export_normalized_graph(self, file_path: str):
        try:
            if not self._normalized:
                raise Exception
            with open(file_path, 'w') as f:
                json.dump(self._data, f)
        except Exception as e:
            print(e)
