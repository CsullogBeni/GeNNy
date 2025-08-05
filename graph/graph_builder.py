import os
import json
import networkx as nx


class GraphBuilder:
    """
    A class for building a graph from a JSON file. Used for saving and loading P4 ASTs as directed graphs.

    Attributes:
        graph (nx.DiGraph): The graph object that stores the nodes and edges.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def set_graph(self, g: nx.DiGraph) -> None:
        """
        Set the graph object.

        Args:
             g (nx.DiGraph): The graph object to set.
        """
        self.graph = g

    def load_data(self, file_path: str) -> None:
        """
        Load graph data from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Raises:
            ValueError: If the file does not exist, is not a file, or is not a JSON file.
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist.")
        if not os.path.isfile(file_path):
            raise ValueError(f"Path {file_path} is not a file.")
        if not file_path.endswith('.json'):
            raise ValueError(f"File {file_path} is not a JSON file.")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        nodes = data.get('nodes', [])
        for node in nodes:
            node_id = node['nodeId']
            attributes = {
                'label': str(node.get('label', '')),
                'line': int(node.get('line', 0)),
                'start': int(node.get('start', 0)),
                'end': int(node.get('end', 0)),
                'value': node.get('value') if node.get('value') is not None else None,
                'nodeId': int(node.get('nodeId', 0)),
                'class_': str(node.get('class_', ''))
            }
            if 'nodeId' == 0 and 'class_' == '':
                continue

            self.graph.add_node(node_id, **attributes)

        edges = data.get('edges', [])
        for edge in edges:
            if int(edge['target']) > int(edge['source']):
                source = int(edge['source'])
                target = int(edge['target'])
            else:
                source = int(edge['target'])
                target = int(edge['source'])

            self.graph.add_edge(source, target)

    def _to_dict(self) -> dict:
        """
        Convert the graph to a dictionary representation.

        Returns:
            dict: A dictionary representation of the graph.
        """
        data = {
            'nodes': [],
            'edges': []
        }

        for node_id, attrs in self.graph.nodes(data=True):
            node_data = {
                'id': int(attrs.get('id', node_id)),
                'label': str(attrs.get('label', '')),
                'line': int(attrs.get('line', 0)),
                'start': int(attrs.get('start', 0)),
                'end': int(attrs.get('end', 0)),
                'value': attrs.get('value', None),
                'nodeId': int(attrs.get('nodeId', 0)),
                'class_': str(attrs.get('class_', ''))
            }
            if node_data['nodeId'] == 0 and node_data['class_'] == '':
                continue
            data['nodes'].append(node_data)

        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {
                'source': int(source),
                'target': int(target)
            }
            data['edges'].append(edge_data)

        return data

    def save_to_json(self, output_path: str, force: bool = True) -> None:
        """
        Save the graph data to a JSON file.

        Args:
            output_path (str): The path to the output JSON file.
            force (bool, optional): If True, overwrite the file if it already exists. Defaults to True.
        """
        if os.path.exists(output_path) and not force:
            raise ValueError(f"File {output_path} already exists. Use force=True to overwrite.")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self._to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def compare_graphs(graph1: nx.DiGraph, graph2: nx.DiGraph) -> bool:
        """
        Compare two graphs, node attributes and edges must be the same.

        Args:
            graph1 (nx.DiGraph): The first graph to compare.
            graph2 (nx.DiGraph): The second graph to compare.

        Returns:
            bool: True if the graphs are the same, False otherwise.
        """
        different = False
        if set(graph1.nodes) != set(graph2.nodes):
            print(set(graph1.nodes), set(graph2.nodes))
            different = True

        for node in graph1.nodes:
            if node not in graph2.nodes:
                print(node)
                different = True

            attrs1 = graph1.nodes[node]
            attrs2 = graph2.nodes[node]

            if attrs1 != attrs2:
                print("node:\n\t", attrs1, "\n\t", attrs2)
                different = True

        if set(graph1.edges) != set(graph2.edges):
            print(set(graph1.edges), set(graph2.edges))
            different = True

        for edge in graph1.edges:
            try:
                attrs1 = graph1.edges[edge]
                attrs2 = graph2.edges[edge]
            except:
                print(edge)
                different = True
                continue

            if attrs1 != attrs2:
                print(attrs1, attrs2)
                different = True
        if different:
            return False
        return True

    def extract_subgraph_by_node_id(self, node_id_value: int) -> nx.DiGraph:
        """
        Find the subgraph that includes the node with the given node_id_value. Clones the graph's subgraph which
        includes the node as root.

        Args:
            node_id_value (int): The node_id_value of the node to extract the subgraph from.

        Returns:
            nx.DiGraph: The subgraph that includes the node with the given node_id_value.
        """
        matching_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('nodeId') == node_id_value]

        if not matching_nodes:
            raise ValueError(f"No node found with noe_id {node_id_value}")

        start_node = matching_nodes[0]

        visited = set()
        queue = [start_node]

        while queue:
            current = queue.pop(0)
            current = self.graph.nodes[current]['nodeId']

            if current in visited:
                continue

            visited.add(current)

            out_edges = []
            for edge in self.graph.edges:
                if edge[0] == current:
                    out_edges.append(edge[1])
            if not out_edges:
                continue

            children = []
            for target in out_edges:
                for node in self.graph.nodes:
                    try:
                        if self.graph.nodes[node]['nodeId'] == target:
                            children.append(target)
                            continue
                    except:
                        continue

            queue.extend(children)

        subgraph = self.graph.subgraph(visited).copy()

        return subgraph

    def find_class(self, class_value: str) -> list:
        """
        Find all nodes with the given class value.

        Args:
            class_value (str): The class value to search for.

        Returns:
            list: A list of node IDs that have the given class value.
        """
        node_ids = []
        for node, attr in self.graph.nodes(data=True):
            if attr.get('class_') == class_value:
                node_ids.append(node)
        return node_ids
