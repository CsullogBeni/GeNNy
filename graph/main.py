import os

from graph_builder import GraphBuilder
from graph_normalizer import GraphNormalizer

# CONFIG
RAW_GRAPH_PATH = "raw_graphs"                   # Path to the raw graphs
NORMALIZED_GRAPH_PATH = "normalized_graphs"     # Path to the normalized graphs to be saved


def main() -> None:
    """
    Normalizes all graphs in the RAW_GRAPH_PATH folder and saves them to the
    NORMALIZED_GRAPH_PATH folder.
    """
    with open("resources/message.txt", "r") as f:
        print(f.read())

    for filename in os.listdir(RAW_GRAPH_PATH):
        print(f"Normalizing {filename}")
        path = os.path.join(RAW_GRAPH_PATH, filename)
        normalizer = GraphNormalizer(path)
        normalizer.normalize()
        normalizer.export_normalized_graph(os.path.join(NORMALIZED_GRAPH_PATH, filename))
        print(f"Normalized {filename}")

    for filename in os.listdir(NORMALIZED_GRAPH_PATH):
        print(f"Saving {filename}")
        path = os.path.join(NORMALIZED_GRAPH_PATH, filename)
        builder = GraphBuilder()
        builder.load_data(path)
        builder.save_to_json(os.path.join(NORMALIZED_GRAPH_PATH, filename))
        print(f"Saved {filename}")

    print("Done")


if __name__ == "__main__":
    main()
