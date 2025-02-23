import networkx as nx
import matplotlib.pyplot as plt

from GrapLearner.p4_graph_data_access import P4GraphDataAccess


class P4Graph:
    def __init__(self):
        self._g = nx.DiGraph()
        self._graph_reader = P4GraphDataAccess()

    @property
    def get_graph(self):
        return self._g

    def add_node(self, node_id: int, line: int, start: int, end: int, class_: str, value: str | None):
        for node in self._g.nodes:
            if self._g.nodes[node]["nodeId"] == node_id:
                raise ValueError(f"Node with id {node_id} already exists in the graph.")
        self._g.add_node(str(node_id), nodeId=node_id, line=line, start=start, end=end, class_=class_, value=value)

    def add_edge(self, source: int, target: int):
        if source not in self._g.nodes or target not in self._g.nodes:
            raise ValueError(f"Node with id {source} or {target} does not exist in the graph.")
        if self._g.has_edge(str(source), str(target)):
            raise ValueError(f"Edge from {source} to {target} already exists in the graph.")
        self._g.add_edge(str(source), str(target))

    def show_graph(self):
        pos = nx.spring_layout(self._g)

        node_labels = {
            node: f"NodeId: {self._g.nodes[node]['nodeId']}\nLine: {self._g.nodes[node]['line']}\nStart: {self._g.nodes[node]['start']}\nEnd: {self._g.nodes[node]['end']}\nClass: {self._g.nodes[node]['class_']}\nValue: {self._g.nodes[node]['value'] if self._g.nodes[node]['value'] is not None else 'None'}"
            for node in self._g.nodes}

        for node in self._g.nodes:
            x, y = pos[node]
            plt.text(x, y, node_labels[node], ha='center', va='center',
                     bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round'))

        # Draw edges
        nx.draw_networkx_edges(self._g, pos, edge_color='k', arrowsize=100, arrowstyle='->')

        for edge in self._g.edges:
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            plt.plot([x1, x2], [y1, y2], 'k-')

        plt.axis('off')
        plt.show()

    def read_from_json(self, file_path):
        self._graph_reader.read_from_json(file_path)
        self._g = self._graph_reader.get_graph

    def read_from_pkl(self, file_path):
        self._graph_reader.read_from_pkl(file_path)
        self._g = self._graph_reader.get_graph

    def write_to_json(self, file_path):
        self._graph_reader.write_to_json(file_path)

    def write_to_pkl(self, file_path):
        self._graph_reader.write_to_pkl(file_path)


graph = P4Graph()
graph.read_from_json(r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\full_graphs\assignment.json")
'''for node_id, attrs in graph.get_graph.nodes(data=True):
    print(f"NodeId: {attrs['nodeId']}, Line: {attrs['line']}, Start: {attrs['start']}, End: {attrs['end']}, Class: {attrs['class_']}, Value: {attrs['value']}")
print(graph.get_graph.edges)'''
graph.show_graph()
