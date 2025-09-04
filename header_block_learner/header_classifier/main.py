"""
Entry point for training and inference of header classifiers over AST graphs.

This module provides a simple CLI-style workflow with two modes:

1) **Training mode** (`TRAIN_MODE = True`)
   - Collects JSON graphs from `DATA_DIR`.
   - Trains and saves two GNN-based classifiers via
     `train_headers.train_and_save_models`:
       • `HeaderBlockClassifier` — predicts header *block* roots/contexts.
       • `HeaderFieldClassifier` — predicts `StructFieldContext` nodes (fields)
         inside header subtrees.

2) **Inference/validation mode** (`TRAIN_MODE = False`)
   - Loads both trained models from `MODEL_DIR`.
   - Accepts a single JSON file or a directory (`CLASSIFY_GRAPH_PATH`).
   - For each graph:
       • Encodes with each model's own encoders.
       • Applies thresholds and type filters
         (`HeaderTypeDeclarationContext` and `StructFieldContext`).
       • Prints readable summaries and top-k diagnostics.

Input JSON format (per graph):
{
  "nodes": [{"id": 1, "class_": "SomeContext", "value": null}, ...],
  "edges": [{"source": 1, "target": 2}, ...]
}

Also supported: a file that contains `{ "graphs": [ ... ] }` or a JSON list of
per-graph objects in the same shape as above.

Configuration constants
-----------------------
- `TRAIN_MODE`: training vs. inference.
- `CLASSIFY_GRAPH_PATH`: file or directory to classify when not training.
- `DATA_DIR`: directory with training JSONs.
- `MODEL_DIR`: where trained models are stored/loaded from.
- `EPOCHS`: number of epochs for training.
- `THRESHOLD_BLOCK`: probability threshold for header block detection.
- `THRESHOLD_FIELD`: probability threshold for header field detection.

Notes
-----
- Each classifier carries its own categorical encoders; graphs must be
  converted separately for each model.
- A couple of hard-coded node IDs (`3273`, `3283`) are used for targeted
  debug prints; feel free to remove or change them.

See also:
- `train_headers.py` for the training orchestration.
- `header_block_classifier.py` and `header_field_classifier.py` for model details.
- `validate_headers.py` for rule-based diagnostics.
"""

# main.py

import os
import json
import torch
import networkx as nx

from train_headers import train_and_save_models
from header_block_classifier import HeaderBlockClassifier
from header_field_classifier import HeaderFieldClassifier

# CONFIGURATION
TRAIN_MODE = False
CLASSIFY_GRAPH_PATH = "validation_data"  # can be a directory or a single file
DATA_DIR = "data"
MODEL_DIR = "models"
EPOCHS = 60
THRESHOLD_BLOCK = 0.95
THRESHOLD_FIELD = 0.95


# --------- utilities ---------
def detect_device() -> str:
    """
    Detect the best available computation device.

    Returns:
        str: "cuda" if a CUDA-capable GPU is available via PyTorch,
             otherwise "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _graph_dict_to_nx(gdict: dict) -> nx.DiGraph:
    """
    Convert a JSON-like graph dictionary to a NetworkX directed graph.

    The input is expected to contain:
      - "nodes": a list of node dicts with at least an "id" key
                 (optionally including "class_", "value", etc.).
      - "edges": a list of edge dicts with "source" and "target" keys.

    Args:
        gdict (dict): Parsed JSON object representing a single graph.

    Returns:
        nx.DiGraph: The constructed directed graph with node attributes
        preserved from the input dictionaries.
    """
    G = nx.DiGraph()
    for node in gdict.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in gdict.get("edges", []):
        src, dst = edge.get("source"), edge.get("target")
        if src in G and dst in G:
            G.add_edge(src, dst)
    return G


def load_graphs_any(path: str) -> list[tuple[str, nx.DiGraph]]:
    """
    Load one or more graphs from a file or a directory.

    Supported file payloads:
      1) A single graph object with "nodes"/"edges".
      2) An object with a "graphs" list, each an individual graph dict.
      3) A top-level JSON list of graph dicts.

    If `path` is a directory, all `*.json` files inside it are scanned.

    Args:
        path (str): Path to a JSON file or a directory containing JSON files.

    Returns:
        list[tuple[str, nx.DiGraph]]: A list of (name, graph) tuples where
        `name` is a stable identifier derived from the filename and index.
    """
    items: list[tuple[str, nx.DiGraph]] = []
    if os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith(".json"):
                fpath = os.path.join(path, fname)
                items.extend(load_graphs_any(fpath))
        return items

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base = os.path.basename(path)
    if isinstance(data, dict) and "nodes" in data:
        items.append((base, _graph_dict_to_nx(data)))
    elif isinstance(data, dict) and "graphs" in data and isinstance(data["graphs"], list):
        for i, gd in enumerate(data["graphs"]):
            name = f"{base}::graph{i}"
            items.append((name, _graph_dict_to_nx(gd)))
    elif isinstance(data, list):
        for i, gd in enumerate(data):
            name = f"{base}::graph{i}"
            if isinstance(gd, dict) and "nodes" in gd:
                items.append((name, _graph_dict_to_nx(gd)))
    else:
        print(f"Unknown graph format: {base}")
    return items


def print_predictions(title: str, indices: list[int], graph: nx.DiGraph, threshold: float) -> None:
    """
    Pretty-print a preview of predicted node matches for a graph.

    Args:
        title (str): Heading to print above the results.
        indices (list[int]): Indices (into the node list ordering) that
            surpassed the relevant threshold.
        graph (nx.DiGraph): The source graph.
        threshold (float): The probability cutoff used for the predictions.

    Returns:
        None: This function prints to stdout for human inspection.
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


def _topk_debug(scores: torch.Tensor, graph: nx.DiGraph, k: int = 10):
    """
    Extract top-k node scores for quick debugging.

    Args:
        scores (torch.Tensor): Per-node scores (1D tensor).
        graph (nx.DiGraph): The corresponding graph whose nodes align with
            the `scores` ordering.
        k (int, optional): Maximum number of entries to return. Defaults to 10.

    Returns:
        list[tuple[int, float, int, str | None, str | None]]:
            A list of tuples (rank, score, node_id, class_, value).
    """
    if scores.numel() == 0:
        return []
    vals, idxs = torch.topk(scores, k=min(k, scores.numel()))
    idxs = idxs.tolist()
    vals = vals.tolist()
    nodes = list(graph.nodes(data=True))
    out = []
    for rank, (i, s) in enumerate(zip(idxs, vals), 1):
        node_id, attrs = nodes[i]
        out.append((rank, float(s), node_id, attrs.get("class_"), attrs.get("value")))
    return out


def _print_node_score(graph: nx.DiGraph, node_id: int, scores: torch.Tensor) -> None:
    """
    Print the model score for a specific node ID, if present in the graph.

    Args:
        graph (nx.DiGraph): Source graph.
        node_id (int): Node identifier to probe.
        scores (torch.Tensor): Per-node scores aligned to the graph's node list.

    Returns:
        None: Prints a single-line diagnostic (or a 'not found' notice).
    """
    node_ids = list(graph.nodes)
    try:
        idx = node_ids.index(node_id)
        sc = float(scores[idx])
        print(f"     score[{node_id}] ({graph.nodes[node_id].get('class_')}): {sc:.4f}")
    except ValueError:
        print(f"     node {node_id} not found in this graph.")


# --------- prediction ---------
@torch.no_grad()
def classify_graph(graph: nx.DiGraph, name: str, model_dir: str, device: str) -> None:
    """
    Classify header blocks and fields for a single graph and print diagnostics.

    Steps:
      1) Load both trained models from `model_dir`.
      2) Build model-specific PyG inputs.
      3) Predict header blocks, filter to `HeaderTypeDeclarationContext`, and
         print matches and top-k candidates.
      4) Predict header fields within the predicted header subtrees, filter to
         `StructFieldContext`, and print matches and top-k candidates.

    Args:
        graph (nx.DiGraph): The input AST graph.
        name (str): A label used in the console output.
        model_dir (str): Directory containing the saved model weights and
            encoders for both classifiers.
        device (str): "cpu" or "cuda".

    Returns:
        None: Outputs are printed for human inspection.
    """
    print(f"\nGraph: {name}")

    # Load models
    block_model = HeaderBlockClassifier(device=device)
    field_model = HeaderFieldClassifier(device=device)
    block_model.load_model(model_dir)
    field_model.load_model(model_dir)

    # Quick GT counts (rule-based) for sanity checks
    gt_block = int(block_model._label_nodes(graph).sum().item())
    gt_field = int(field_model._label_nodes(graph).sum().item())
    print(f"   GT headers: {gt_block} | GT fields: {gt_field}")

    # Separate PyG inputs for both models — each has its own encoders
    pyg_block = block_model._graph_to_pyg(graph).to(device)
    pyg_field = field_model._graph_to_pyg(graph).to(device)

    node_ids = list(graph.nodes)

    # --- HeaderBlockClassifier ---
    block_model.eval()
    emb_block = block_model.gnn(pyg_block.x, pyg_block.edge_index)
    logits_block = block_model.head(emb_block).squeeze(-1)
    scores_block = torch.sigmoid(logits_block)

    # Thresholding + TYPE FILTER: only HeaderTypeDeclarationContext
    raw_block_hits = [i for i, s in enumerate(scores_block.tolist()) if s > THRESHOLD_BLOCK]
    block_hits = [i for i in raw_block_hits
                  if graph.nodes[node_ids[i]].get("class_") == "HeaderTypeDeclarationContext"]

    print_predictions("HeaderBlockClassifier (HeaderTypeDeclarationContext)", block_hits, graph, THRESHOLD_BLOCK)

    # Focused debug for a specific header node id (example: 3273)
    _print_node_score(graph, 3273, scores_block)
    if not block_hits:
        top = _topk_debug(scores_block, graph, k=10)
        if top:
            print("HeaderBlock top-10 candidates (score, nodeId, class, value):")
            for r, s, nid, c, v in top:
                print(f"   {r:2d}. {s:0.4f} | id={nid} | class={c} | value={v}")

    # --- HeaderFieldClassifier ---
    field_model.eval()
    emb_field = field_model.gnn(pyg_field.x, pyg_field.edge_index)
    logits_field = field_model.head(emb_field).squeeze(-1)
    scores_field = torch.sigmoid(logits_field)

    # Candidate set: descendants of block-predicted header roots
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
        cand_idx = list(range(len(node_ids)))  # fallback: full graph

    # Threshold + TYPE FILTER within candidates: only StructFieldContext
    field_hits = [i for i in cand_idx
                  if scores_field[i] > THRESHOLD_FIELD
                  and graph.nodes[node_ids[i]].get("class_") == "StructFieldContext"]

    print_predictions("📐 HeaderFieldClassifier (StructFieldContext)", field_hits, graph, THRESHOLD_FIELD)

    # Focused debug for a specific field node id (example: 3283)
    _print_node_score(graph, 3283, scores_field)

    if not field_hits:
        # top-10 within candidates, ONLY StructFieldContext
        pairs = [(i, float(scores_field[i])) for i in cand_idx
                 if graph.nodes[node_ids[i]].get("class_") == "StructFieldContext"]
        pairs.sort(key=lambda t: t[1], reverse=True)
        top_pairs = pairs[:10]
        if top_pairs:
            print("HeaderField top-10 StructFieldContext (within header subtree only):")
            for rank, (i, sc) in enumerate(top_pairs, 1):
                nid = node_ids[i]
                print(f"   {rank:2d}. {sc:0.4f} | id={nid} | class=StructFieldContext")


def run_predictions(path: str, model_dir: str, device: str) -> None:
    """
    Load graphs from `path` and run classification for each graph.

    Args:
        path (str): File or directory pointing to validation data.
        model_dir (str): Directory containing saved models/encoders.
        device (str): "cpu" or "cuda".

    Returns:
        None: Progress and results are printed to stdout.
    """
    graphs = load_graphs_any(path)
    if not graphs:
        print(f"No graphs found at: {path}")
        return
    print(f"{len(graphs)} graph(s) loaded from '{path}'.")
    for name, G in graphs:
        classify_graph(G, name, model_dir, device)


# --------- main ---------
def main() -> None:
    """
    Program entry point.

    Behavior depends on `TRAIN_MODE`:
      - If True, trains both models on data from `DATA_DIR` and saves to
        `MODEL_DIR` for `EPOCHS` epochs.
      - If False, loads models from `MODEL_DIR` and classifies graphs found at
        `CLASSIFY_GRAPH_PATH`.

    Returns:
        None
    """
    device = detect_device()
    print(f"Device: {device.upper()}")

    if TRAIN_MODE:
        print("   Training mode is ON.")
        print(f"  - Data: {DATA_DIR}")
        print(f"  - Models: {MODEL_DIR}")
        print(f"  - Epochs: {EPOCHS}")
        train_and_save_models(data_dir=DATA_DIR, output_dir=MODEL_DIR, epochs=EPOCHS)
        print("Training complete. Models saved.")
    else:
        if not os.path.isdir(MODEL_DIR):
            print(f"Model directory not found: {MODEL_DIR}")
            return
        run_predictions(CLASSIFY_GRAPH_PATH, MODEL_DIR, device=device)
        print("\nValidation/prediction complete.")


if __name__ == "__main__":
    main()
