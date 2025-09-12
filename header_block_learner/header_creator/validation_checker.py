import os

from graph.graph_builder import GraphBuilder
from graph.graph_visualizer import GraphVisualizer

from pretty_printer.pretty_printer import PrettyPrinter

val_data = "validation_out"

for file in os.listdir(val_data):
    graph_builder = GraphBuilder()
    graph_builder.load_data(os.path.join(val_data, file))
    pretty_printer = PrettyPrinter(graph_builder.graph)
    print(file + ':\n' + pretty_printer.get_script)
    out_path = os.path.join("generated_p4_files", file[:-5] + '.p4')
    pretty_printer.save_script(out_path)
    print('Saved to: ' + out_path)
    graph_visualizer = GraphVisualizer()
    graph_visualizer.draw(graph_builder.graph)
