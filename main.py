import json
import os
from typing import Dict, Union, List

import networkx as nx
import torch

from graph.graph_builder import GraphBuilder
from graph.graph_normalizer import GraphNormalizer
from empty_else_detector.empty_else_detector import EmptyElseDetector
from header_block_learner.header_classifier.header_block_classifier import HeaderBlockClassifier
from header_block_learner.header_classifier.header_field_classifier import HeaderFieldClassifier
from header_block_learner.header_creator.header_completion_model import HeaderCompletionModel


def _normalize_graphs(path: str) -> None:
    """
    Normalize and rebuild a graph JSON file in a canonical format.
    The resulting normalized graph overwrites the intermediate file.

    Args:
        path (str): Path to the input graph JSON file.

    """
    normalizer = GraphNormalizer(path)
    normalizer.normalize()
    path = "p4_recources\\normalized_graph.json"
    normalizer.export_normalized_graph(path)

    builder = GraphBuilder()
    builder.load_data(path)
    builder.save_to_json(path)


def _evaluate_empty_else_detector(path: str) -> None:
    """
    Run inference with the EmptyElseDetector model on a graph and print results.

    The function:
      - Loads a pretrained EmptyElseDetector model.
      - Encodes the graph into node embeddings.
      - Predicts indices corresponding to empty else blocks.
      - Maps predicted indices back to graph nodes.
      - Prints a human-readable summary of detected empty else blocks.

    Args:
        path (str): Path to the normalized graph JSON file.
    """
    print('\n========== Empty Else Detector ===========\n')
    detector = EmptyElseDetector()
    detector = EmptyElseDetector(device="cpu")
    detector.load_model("empty_else_detector")

    with torch.no_grad():
        node_emb = detector.encode_graph(path)
        pred_idx = detector.predict_subgraph(path, node_emb)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes_in_order = data.get("nodes", [])
    hits = []
    for i in pred_idx:
        if 0 <= i < len(nodes_in_order):
            n = nodes_in_order[i]
            hits.append({
                "index": i,
                "id": n.get("id", n.get("nodeId")),
                "class_": n.get("class_"),
                "value": n.get("value"),
            })

    if not hits:
        print("[Model] No empty else block found")
    else:
        print(f"[Model] {len(hits)} classified empty else block(s)+:")
        for h in hits:
            print(f"  - idx={h['index']} id={h['id']} class_={h['class_']} value={h['value']}")

    print('\n========== Empty Else Detector ===========\n')


@torch.no_grad()
def _evaluate_header_block_classifier(path: str) -> None:
    """
    Evaluate header block and header field classifiers on a graph.

    This function performs a full inference pipeline:
      - Loads the graph into a NetworkX DiGraph.
      - Loads pretrained HeaderBlockClassifier and HeaderFieldClassifier models.
      - Computes ground-truth statistics for debugging purposes.
      - Runs GNN inference to score header blocks and fields.
      - Applies probability thresholds to select predictions.
      - Restricts field predictions to subtrees of predicted headers.
      - Prints formatted prediction summaries.

    Args:
        path (str): Path to the normalized graph JSON file.
    """
    print('\n========== Header Block Classifier ===========\n')

    graph = _graph_to_nx(path)
    THRESHOLD_BLOCK = 0.95
    THRESHOLD_FIELD = 0.5

    block_model = HeaderBlockClassifier(device='cpu')
    field_model = HeaderFieldClassifier(device='cpu')
    block_model.load_model('header_block_learner\\header_classifier\models')
    field_model.load_model('header_block_learner\\header_classifier\\models')

    gt_block = int(block_model._label_nodes(graph).sum().item())
    gt_field = int(field_model._label_nodes(graph).sum().item())
    print(f"   GT headers: {gt_block} | GT fields: {gt_field}")

    pyg_block = block_model._graph_to_pyg(graph).to('cpu')
    pyg_field = field_model._graph_to_pyg(graph).to('cpu')

    node_ids = list(graph.nodes)

    block_model.eval()
    emb_block = block_model.gnn(pyg_block.x, pyg_block.edge_index)
    logits_block = block_model.head(emb_block).squeeze(-1)
    scores_block = torch.sigmoid(logits_block)

    raw_block_hits = [i for i, s in enumerate(scores_block.tolist()) if s > THRESHOLD_BLOCK]
    block_hits = [i for i in raw_block_hits
                  if graph.nodes[node_ids[i]].get("class_") == "HeaderTypeDeclarationContext"]

    print_predictions("HeaderBlockClassifier (HeaderTypeDeclarationContext)", block_hits, graph, THRESHOLD_BLOCK)

    field_model.eval()
    emb_field = field_model.gnn(pyg_field.x, pyg_field.edge_index)
    logits_field = field_model.head(emb_field).squeeze(-1)
    scores_field = torch.sigmoid(logits_field)

    pred_headers = [node_ids[i] for i in block_hits]
    cand = set()
    for h in pred_headers:
        q, vis = [h], set([h])
        while q:
            cur = q.pop(0)
            cand.add(cur)
            for nxt in graph.successors(cur):
                if nxt not in vis:
                    vis.add(nxt)
                    q.append(nxt)

    if cand:
        cand_idx = [i for i, nid in enumerate(node_ids) if nid in cand]
    else:
        cand_idx = list(range(len(node_ids)))

    field_hits = [i for i in cand_idx
                  if scores_field[i] > THRESHOLD_FIELD
                  and graph.nodes[node_ids[i]].get("class_") == "StructFieldContext"]

    print_predictions("HeaderFieldClassifier (StructFieldContext)", field_hits, graph, THRESHOLD_FIELD)

    print('\n========== Header Block Classifier ===========\n')


def _graph_to_nx(path: str) -> nx.DiGraph:
    """
    Load a graph JSON file into a NetworkX directed graph.

    The JSON file is expected to contain:
      - A "nodes" list with node attributes (including "id").
      - An "edges" list with "source" and "target" identifiers.

    Invalid edges (referencing missing nodes) are ignored.

    Args:
        path (str): Path to the graph JSON file.

    Returns:
        nx.DiGraph: A directed graph with node attributes preserved.
    """
    with open(path, "r") as f:
        gdict = json.load(f)

    G = nx.DiGraph()
    for node in gdict.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in gdict.get("edges", []):
        src, dst = edge.get("source"), edge.get("target")
        if src in G and dst in G:
            G.add_edge(src, dst)
    return G


def print_predictions(title: str, indices: list[int], graph: nx.DiGraph, threshold: float) -> None:
    """
    Pretty-print a preview of predicted node matches for a graph.

    This helper prints a human-readable summary of nodes whose prediction
    probability exceeded a given threshold. The output includes:
      - Node index in the model's ordering
      - Node ID
      - Node class
      - Optional terminal value

    Only a limited number of results are printed for readability.

    Args:
        title (str): Heading to print above the results.
        indices (list[int]): Indices of predicted nodes.
        graph (nx.DiGraph): Source graph containing node attributes.
        threshold (float): Probability threshold used for prediction.
    """
    node_ids = list(graph.nodes)
    print(f"\n{title} | matches: {len(indices)}  (threshold: {threshold})")
    preview_max = 20
    for i, idx in enumerate(indices[:preview_max], start=1):
        nid = node_ids[idx] if 0 <= idx < len(node_ids) else f"<idx:{idx}>"
        attrs = graph.nodes[nid] if nid in graph else {}
        klass = attrs.get("class_", "?")
        val = attrs.get("value", None)
        if val is not None:
            print(f"  {i:>2}. idx={idx:<5} id={nid}  class_={klass}  value={val}")
        else:
            print(f"  {i:>2}. idx={idx:<5} id={nid}  class_={klass}")
    if len(indices) > preview_max:
        print(f"  ... and {len(indices) - preview_max} more.")


def _evaluate_header_block_creator(path: str) -> None:
    """
    Run header completion using a predefined configuration file.

    This function applies the HeaderCompletionModel to insert missing
    header fields into the graph based on a configuration JSON file.
    The completed graph is written to disk and the process is logged.

    Args:
        path (str): Path to the normalized graph JSON file.
    """
    print('\n========== Header Block Creator ===========\n')

    out = complete_graph_from_config(
        graph_path=path,
        config_path='header_block_creator_config.json',
        model_dir='header_block_learner//header_creator//models',
        output_path='header_creator_output.json',
    )

    print('\n========== Header Block Creator ===========\n')


def complete_graph_from_config(
        graph_path: str,
        config_path: str,
        model_dir: str,
        output_path: str | None = None,
        device: str | None = None,
) -> str:
    """
    Complete a graph by inserting header fields defined in a configuration file.

    This function:
      - Loads a pretrained HeaderCompletionModel.
      - Reads header/field insertion rules from a JSON config.
      - Applies deterministic AST insertions using the model.
      - Logs which headers and fields were actually added.

    Header targets may be specified either by:
      - Header node ID (int)
      - Header name (str)

    Args:
        graph_path (str): Path to the input graph JSON file.
        config_path (str): Path to the header insertion configuration JSON.
        model_dir (str): Directory containing the trained model checkpoint.
        output_path (str | None): Optional output path for the completed graph.
        device (str | None): Torch device ("cpu" or "cuda"). Auto-detected if None.

    Returns:
        str: Path to the written completed graph JSON file.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = HeaderCompletionModel(device=device)
    model.load_model(model_dir)

    additions = load_config(config_path)
    G_before = model.load_graph_json(graph_path)

    out_path = model.complete_graph(
        graph_path=graph_path,
        additions=additions,
        output_path=output_path,
    )

    G_after = model.load_graph_json(out_path)

    print("\nAfter insertions:")

    for hdr, fields in additions.items():
        if isinstance(hdr, int):
            header_ids = [hdr] if hdr in G_after else []
        else:
            header_ids = HeaderCompletionModel._find_header_by_name(G_after, hdr)

        for hid in header_ids:
            hname = get_header_name(G_after, hid) or "<unknown>"
            print(f"\n  -> Header: id={hid}, name={hname}")

            before_fields = set(
                HeaderCompletionModel._struct_fields_under(G_before, hid)
            ) if hid in G_before else set()

            after_fields = set(
                HeaderCompletionModel._struct_fields_under(G_after, hid)
            )

            new_fields = after_fields - before_fields

            if not new_fields:
                print("     (Insertions have no effect)")
                continue

            for fid in new_fields:
                spec = HeaderCompletionModel._extract_field_tokens(G_after, fid)
                print(
                    f"     + field: type={spec.type}, name={spec.name}"
                )


def load_config(path: str) -> Dict[Union[int, str], List[dict]]:
    """
    Load a header insertion configuration from a JSON file.

    The configuration maps header identifiers to lists of field specifications.
    Header identifiers that are digit-only strings are converted to integers.

    Example input:
        {
          "ethernet_t": [{"type": "macAddr_t", "name": "dst"}],
          "1234": [{"type": "bit<16>", "name": "len"}]
        }

    Args:
        path (str): Path to the configuration JSON file.

    Returns:
        Dict[Union[int, str], List[dict]]: Normalized header-to-fields mapping.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[Union[int, str], List[dict]] = {}
    for k, v in raw.items():
        key = int(k) if str(k).isdigit() else k
        out[key] = list(v)
    return out


def get_header_name(G: nx.DiGraph, header_id: int) -> str | None:
    """
    Extract the textual name of a header from its AST subtree.

    The function performs a breadth-first search under the given header node
    and returns the first terminal token consistent with a header declaration.

    Args:
        G (nx.DiGraph): AST graph.
        header_id (int): Node ID of the header declaration.

    Returns:
        str | None: Header name if found, otherwise None.
    """
    return HeaderCompletionModel._first_value_under(
        G,
        header_id,
        class_path_prefix=("HeaderTypeDeclarationContext",)
    )


def main() -> None:
    """
    End-to-end evaluation pipeline for graph-based P4 analysis.

    The pipeline performs:
      1. Graph normalization and rebuilding.
      2. Empty else block detection.
      3. Header block and field classification.
      4. Header field completion based on learned models.

    This function serves as the main entry point for manual evaluation
    and debugging of the full system.
    """
    path = "p4_recources\graph_output.json"

    _normalize_graphs(path)
    path = "p4_recources\\normalized_graph.json"

    _evaluate_empty_else_detector(path)
    _evaluate_header_block_classifier(path)
    _evaluate_header_block_creator(path)


if __name__ == "__main__":
    main()
