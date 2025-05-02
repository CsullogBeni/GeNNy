import json
import os
from copy import deepcopy

BASE_PATH = "data/basic_p4_v2_normalized.json"
OUTPUT_DIR = "data/"

# List of renamings to apply (old_name, new_name)
RENAMING_RULES = [
    ("egress_spec", "egress_specific"),
    ("dstAddr", "destinationAddress"),
    ("srcAddr", "sourceAddress"),
    ("etherType", "ethernetType"),

]


def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_graph(graph, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


def apply_renaming(graph, old_name, new_name):
    modified = deepcopy(graph)
    count = 0
    for node in modified["nodes"]:
        if node.get("value") == old_name:
            node["value"] = new_name
            count += 1
    return modified, count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_graph = load_graph(BASE_PATH)
    print(f"Loaded base graph with {len(base_graph['nodes'])} nodes.")

    for old_name, new_name in RENAMING_RULES:
        renamed_graph, count = apply_renaming(base_graph, old_name, new_name)
        out_path = os.path.join(OUTPUT_DIR, f"renamed_{old_name}_to_{new_name}.json")
        save_graph(renamed_graph, out_path)
        print(f"Saved: {out_path} (Renamed {count} occurrences)")


if __name__ == "__main__":
    main()
