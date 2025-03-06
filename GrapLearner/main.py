from GrapLearner.model import Model
from GrapLearner.p4_graph import P4Graph

if __name__ == '__main__':
    model = Model()
    model.load(r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\2025-03-06-11-45-44model.pth")
    # model.add_graph_from_file(r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\full_graphs\assignment.json")
    # model.add_graph_from_file(r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\full_graphs\assignment.json")
    # model.train()
    # model.save()
    og_graph = P4Graph()
    og_graph.read_from_json(r"C:\Users\Acer\OneDrive - Eotvos Lorand Tudomanyegyetem\Dokumentumok\git\P4\GrapLearner\full_graphs\assignment.json")
    differences = model.compare_graphs(og_graph.get_graph)
    print(differences['missing_edges'])
    print(differences['extra_edges'])
    for item in differences['node_attribute_differences']:
        print(f"Node {item['node']}:")
        print('Original:', item['original'])
        print('Reconstructed:', item['reconstructed'])
