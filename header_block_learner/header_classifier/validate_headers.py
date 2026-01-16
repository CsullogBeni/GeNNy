"""
Utilities for validating P4 AST header structures and listing header fields.

This module loads AST graphs from JSON (nodes/edges) into a directed NetworkX graph
and provides two helpers:

- `_is_header_block(graph, node_id) -> bool`:
    Returns True iff `node_id` is the root of a header block according to the
    grammar pattern:
        InputContext
          └─ DeclarationContext
              └─ TypeDeclarationContext
                  └─ DerivedTypeDeclarationContext
                      └─ HeaderTypeDeclarationContext

- `list_struct_field_nodes(graph) -> dict[int, list[int]]`:
    Finds all header blocks in `graph`, then performs a BFS from each
    `HeaderTypeDeclarationContext` to collect `StructFieldContext` nodes reachable
    within that header subtree. Prints a readable summary and returns the mapping
    `{header_root_id: [struct_field_node_ids...]}`.

Expected JSON format (per file):
{
  "nodes": [{"id": 1, "class_": "SomeContext", "value": null}, ...],
  "edges": [{"source": 1, "target": 2}, ...]
}

Notes
-----
- Node order is not important here; we traverse by explicit edges.
- Only nodes with `class_ == "TerminalNodeImpl"` may have a meaningful `value`;
  others typically keep `value: null`.
- All logging uses `print`, no external logging package is required.
"""

import os
import json
from typing import Dict, List

import networkx as nx


def _is_header_block(graph: nx.DiGraph, node_id: int) -> bool:
    """
    Check whether a node is the root of a header block.

    The check follows the grammar chain:
        InputContext
          -> DeclarationContext
          -> TypeDeclarationContext
          -> DerivedTypeDeclarationContext
          -> HeaderTypeDeclarationContext

    Args:
        graph (nx.DiGraph): Directed AST graph with node attributes.
        node_id (int): Candidate node to test.

    Returns:
        bool: True if the required descendant chain exists, False otherwise.
    """
    try:
        if graph.nodes[node_id].get("class_") != "InputContext":
            return False

        for child1 in graph.successors(node_id):
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
                            return True
        return False
    except KeyError:
        # Missing attributes on some nodes — treat as non-header.
        return False
    except Exception:
        # Be defensive: do not break the caller in case of malformed graphs.
        return False


def list_struct_field_nodes(graph: nx.DiGraph) -> Dict[int, List[int]]:
    """
    Locate header blocks and list StructFieldContext nodes in their subtrees.

    For each header root `h` (node where `_is_header_block(graph, h)` is True):
    1) Identify all direct children that are `HeaderTypeDeclarationContext`.
    2) BFS from each such child and collect nodes whose `class_ == "StructFieldContext"`.
    3) Print a readable per-header summary.

    Args:
        graph (nx.DiGraph): Directed AST graph loaded from the JSON format described above.

    Returns:
        dict[int, list[int]]: Mapping from header root ids to lists of
                              StructFieldContext node ids discovered beneath them.
    """
    headers = [n for n in graph.nodes if _is_header_block(graph, n)]
    print(f"Found {len(headers)} header root candidate(s): {headers}")

    results: Dict[int, List[int]] = {}

    for h in headers:
        visited = set()
        struct_fields: List[int] = []
        queue: List[int] = []

        # Enqueue all HeaderTypeDeclarationContext nodes under the grammar chain
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

        # BFS within the header subtree(s)
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

        results[h] = struct_fields

        # Per-header verbose summary
        print(f"\n  Header root {h}")
        print(f"   • Header subtree size (unique nodes visited): {len(visited)}")
        print(f"   • StructFieldContext count: {len(struct_fields)}")
        if struct_fields:
            preview = ", ".join(map(str, struct_fields[:10]))
            more = f" (… +{len(struct_fields) - 10} more)" if len(struct_fields) > 10 else ""
            print(f"   • StructFieldContext node ids (up to 10): {preview}{more}")

    # Global summary
    total_fields = sum(len(v) for v in results.values())
    print("\n  Summary")
    print(f"   • Header roots found: {len(results)}")
    print(f"   • Total StructFieldContext nodes across headers: {total_fields}")

    # Top-headers by field count (for quick diagnostics)
    if results:
        ranking = sorted(results.items(), key=lambda kv: len(kv[1]), reverse=True)
        top = ranking[:5]
        print("   • Top headers by field count:")
        for rank, (hid, fields) in enumerate(top, 1):
            print(f"     {rank}. header {hid}: {len(fields)} field(s)")

    return results


if __name__ == "__main__":
    """
    Script entry-point: scan the local 'data' directory for JSON graphs and
    print header/field diagnostics per file plus an overall per-run summary.
    """
    data_dir = "data"
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        raise SystemExit(1)

    processed = 0
    files_with_headers = 0
    total_headers = 0
    total_fields = 0

    for file in sorted(os.listdir(data_dir)):
        if not file.endswith(".json"):
            continue

        path = os.path.join(data_dir, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"\nSkipping '{file}' (failed to read/parse JSON): {e}")
            continue

        G = nx.DiGraph()
        for node in data.get("nodes", []):
            G.add_node(node["id"], **node)
        for edge in data.get("edges", []):
            G.add_edge(edge["source"], edge["target"])

        print(f"\nFile: {file}")
        res = list_struct_field_nodes(G)

        processed += 1
        total_headers += len(res)
        fields_here = sum(len(v) for v in res.values())
        total_fields += fields_here
        if res:
            files_with_headers += 1

    # Per-run aggregate
    print("\n================ RUN SUMMARY ================")
    print(f"Files processed:                 {processed}")
    print(f"Files containing header roots:   {files_with_headers}")
    print(f"Total header roots:              {total_headers}")
    print(f"Total StructFieldContext nodes:  {total_fields}")
    print("============================================")
