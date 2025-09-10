import os

from graph.graph_builder import GraphBuilder
from pretty_printer.pretty_printer import PrettyPrinter


val_data = "validation"

for file in os.listdir(val_data):
    graph_builder = GraphBuilder()
    graph_builder.load_data(os.path.join(val_data, file))
    pretty_printer = PrettyPrinter(graph_builder.graph)
    print(file + ':\n' + pretty_printer.get_script)
