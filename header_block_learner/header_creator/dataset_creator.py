import os

from graph.graph_builder import GraphBuilder
from pretty_printer.pretty_printer import PrettyPrinter


# FIXME basic_p4_with_new_header_0 to basic_p4_with_new_header_14 contains only one header field

def process_graph(file: str, index: int, header_id: int, id_to_delete_subgraph: int):
    graph_builder = GraphBuilder()
    graph_builder.load_data(file)
    new_graph = graph_builder.extract_subgraph_by_node_id(header_id)
    new_graph_builder = GraphBuilder()
    new_graph_builder.set_graph(new_graph)
    new_graph_builder.save_to_json("data\\basic_p4_with_new_header_" + str(index) + "_contains_only_subgraph.json",
                                   force=True)
    pretty_printer = PrettyPrinter(new_graph)
    print("Original header:\n" + pretty_printer.get_script)
    new_graph_reduced = new_graph_builder.clone_without_subgraph_by_node_id(id_to_delete_subgraph)
    new_graph_reduced_builder = GraphBuilder()
    new_graph_reduced_builder.set_graph(new_graph_reduced)
    new_graph_reduced_builder.save_to_json(
        "data\\basic_p4_with_new_header_" + str(index) + "_contains_only_subgraph_reduced.json",
        force=True)
    pretty_printer = PrettyPrinter(new_graph_reduced)
    print("Reduced header:\n" + pretty_printer.get_script)


data_path = os.path.join(os.path.dirname(os.getcwd()), "header_classifier\\data")
assert os.path.exists(data_path)

for index in range(15):
    file = os.path.join(data_path, "basic_p4_with_new_header_" + str(index) + ".json")
    assert os.path.exists(file)
    print("Processing:", file)
    process_graph(file, index, 3270, 3281)

for index in range(15, 25):
    file = os.path.join(data_path, "basic_p4_with_new_header_" + str(index) + ".json")
    assert os.path.exists(file)
    print("Processing:", file)
    process_graph(file, index, 3271, 3283)

file = os.path.join(data_path, 'basic_p4.json')
assert os.path.exists(file)
process_graph(file, 26, 0, 3283)
process_graph(file, 27, 0, 3310)
process_graph(file, 28, 0, 3346)
process_graph(file, 29, 0, 3345)
process_graph(file, 30, 0, 3344)
process_graph(file, 31, 0, 3343)
process_graph(file, 32, 0, 3342)
process_graph(file, 33, 0, 3341)
process_graph(file, 34, 0, 3340)
process_graph(file, 35, 0, 3339)
process_graph(file, 36, 0, 3338)
process_graph(file, 37, 0, 3337)
process_graph(file, 38, 0, 3336)
process_graph(file, 39, 0, 3490)
