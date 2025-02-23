import json
import networkx as nx
import pickle

from GrapLearner.graph_reader import GraphReader
from GrapLearner.graph_writer import GraphWriter


class P4GraphDataAccess(GraphReader, GraphWriter):
    def __init__(self):
        self._g = nx.DiGraph()

    def read_from_json(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for node in data['nodes']:
                    self._g.add_node(str(node['nodeId']), nodeId=node['nodeId'], line=node['line'], start=node['start'],
                                     end=node['end'], class_=node['class_'], value=node['value'])
                for edge in data['edges']:
                    self._g.add_edge(str(edge['source']), str(edge['target']))
        except Exception as e:
            print("Error: ", e)

    def read_from_pkl(self, file_path):
        try:
            with open("graph.pkl", "rb") as f:
                self._g = pickle.load(f)
        except Exception as e:
            print(e)

    def write_to_json(self, file_path):
        dictionary = {"nodes": [], "edges": []}
        for node in self._g.nodes:
            dictionary["nodes"].append(
                {"nodeId": self._g.nodes[node]["nodeId"],
                 "line": self._g.nodes[node]["line"],
                 "start": self._g.nodes[node]["start"],
                 "end": self._g.nodes[node]["end"],
                 "class_": self._g.nodes[node]["class_"],
                 "value": self._g.nodes[node]["value"]})
        for edge in self._g.edges:
            dictionary["edges"].append(
                {"source": edge[0], "target": edge[1]})
        try:
            with open(file_path, 'w') as f:
                json.dump(dictionary, f)
        except Exception as e:
            print(e)

    def write_to_pkl(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self._g, f)

    @property
    def get_graph(self):
        return self._g


grap_constructor = P4GraphDataAccess()
grap_constructor.read_from_json(r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\full_graphs\assignment.json")
for node_id, attrs in grap_constructor.get_graph.nodes(data=True):
    print(f"NodeId: {attrs['nodeId']}, Line: {attrs['line']}, Start: {attrs['start']}, End: {attrs['end']}, Class: {attrs['class_']}, Value: {attrs['value']}")
print(grap_constructor.get_graph.edges)
'''for inner_node in grap_constructor.get_graph.nodes:
    print(inner_node)'''
grap_constructor.write_to_json(r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\full_graphs\assignment_2.json")
g1 = grap_constructor.get_graph
grap_constructor.read_from_json(r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\full_graphs\assignment_2.json")
g2 = grap_constructor.get_graph
assert g1 == g2
