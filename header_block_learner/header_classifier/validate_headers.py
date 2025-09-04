import os

import networkx as nx
import json


def _is_header_block(graph: nx.DiGraph, node_id: int) -> bool:
    try:
        if graph.nodes[node_id].get("class_") != "InputContext":
            return False

        children = list(graph.successors(node_id))
        for child1 in children:
            if graph.nodes[child1].get("class_") == "DeclarationContext":
                for child2 in graph.successors(child1):
                    if graph.nodes[child2].get("class_") == "TypeDeclarationContext":
                        for child3 in graph.successors(child2):
                            if graph.nodes[child3].get("class_") == "DerivedTypeDeclarationContext":
                                for child4 in graph.successors(child3):
                                    if graph.nodes[child4].get("class_") == "HeaderTypeDeclarationContext":
                                        return True
        return False
    except:
        return False


def list_struct_field_nodes(graph: nx.DiGraph):
    headers = [n for n in graph.nodes if _is_header_block(graph, n)]
    print("Header block node IDs:", headers)

    for h in headers:
        visited = set()
        struct_fields = []

        # BFS a header block leszármazottain, HeaderTypeDeclarationContext-től indulva
        queue = []

        # megkeressük a HeaderTypeDeclarationContext node-okat, amelyek az _is_header_block szerint léteznek
        for child1 in graph.successors(h):
            if graph.nodes[child1].get("class_") != "DeclarationContext":
                continue
            for child2 in graph.successors(child1):
                if graph.nodes[child2].get("class_") != "TypeDeclarationContext":
                    continue
                for child3 in graph.successors(child2):
                    if graph.nodes[child3].get("class_") != "DerivedTypeDeclarationContext":
                        continue
                    for child4 in graph.successors(child3):
                        if graph.nodes[child4].get("class_") == "HeaderTypeDeclarationContext":
                            queue.append(child4)

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if graph.nodes[current].get("class_") == "StructFieldContext":
                struct_fields.append(current)

            for succ in graph.successors(current):
                if succ not in visited:
                    queue.append(succ)

        print(f"\nHeader Block Node {h} - StructFieldContext nodes:")
        for node_id in struct_fields:
            print(f"  Node ID: {node_id}, Class: StructFieldContext")


if __name__ == "__main__":
    for file in os.listdir('data'):
        if file.endswith(".json"):
            path = os.path.join('data', file)

            with open(path, 'r') as f:
                data = json.load(f)

            G = nx.DiGraph()
            for node in data['nodes']:
                G.add_node(node['id'], **node)
            for edge in data.get('edges', []):
                G.add_edge(edge['source'], edge['target'])
            print("file:", file)
            list_struct_field_nodes(G)
            print()
