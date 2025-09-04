# main.py

import os
import json
import torch
import networkx as nx

from train_headers import train_and_save_models
from header_block_classifier import HeaderBlockClassifier
from header_field_classifier import HeaderFieldClassifier

# 🔧 KONFIGURÁCIÓ
TRAIN_MODE = True
CLASSIFY_GRAPH_PATH = "validation_data"  # lehet mappa vagy fájl
DATA_DIR = "data"
MODEL_DIR = "models"
EPOCHS = 60

THRESHOLD_BLOCK = 0.95
THRESHOLD_FIELD = 0.65  # rugalmasabb, a Field tipikusan alacsonyabb score-okat ad


# --------- utilok ---------
def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _graph_dict_to_nx(gdict: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in gdict.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in gdict.get("edges", []):
        src, dst = edge.get("source"), edge.get("target")
        if src in G and dst in G:
            G.add_edge(src, dst)
    return G


def load_graphs_any(path: str) -> list[tuple[str, nx.DiGraph]]:
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
        print(f"⚠️ Nem ismert graf formátum: {base}")
    return items


def print_predictions(title: str, indices: list[int], graph: nx.DiGraph, threshold: float):
    node_ids = list(graph.nodes)
    print(f"\n{title} | találatok: {len(indices)}  (küszöb: {threshold})")
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
        print(f"  ... és még {len(indices) - preview_max} további találat.")


def _topk_debug(scores: torch.Tensor, graph: nx.DiGraph, k: int = 10):
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


def _print_node_score(graph: nx.DiGraph, node_id: int, scores: torch.Tensor):
    node_ids = list(graph.nodes)
    try:
        idx = node_ids.index(node_id)
        sc = float(scores[idx])
        print(f"   🔎 score[{node_id}] ({graph.nodes[node_id].get('class_')}): {sc:.4f}")
    except ValueError:
        print(f"   🔎 node {node_id} not found in this graph.")


# --------- predikció ---------
@torch.no_grad()
def classify_graph(graph: nx.DiGraph, name: str, model_dir: str, device: str):
    print(f"\n🔍 Gráf: {name}")

    # modellek betöltése
    block_model = HeaderBlockClassifier(device=device)
    field_model = HeaderFieldClassifier(device=device)
    block_model.load_model(model_dir)
    field_model.load_model(model_dir)

    # GT (szabály alapú) darabszámok – gyors ellenőrzéshez
    gt_block = int(block_model._label_nodes(graph).sum().item())
    gt_field = int(field_model._label_nodes(graph).sum().item())
    print(f"   GT headers: {gt_block} | GT fields: {gt_field}")

    # ❗ KÜLÖN PyG input mindkét modellhez – külön encoderek!
    pyg_block = block_model._graph_to_pyg(graph).to(device)
    pyg_field = field_model._graph_to_pyg(graph).to(device)

    node_ids = list(graph.nodes)

    # --- HeaderBlockClassifier ---
    block_model.eval()
    emb_block = block_model.gnn(pyg_block.x, pyg_block.edge_index)
    logits_block = block_model.head(emb_block).squeeze(-1)
    scores_block = torch.sigmoid(logits_block)

    # küszöbölés + OSZTÁLYSZŰRÉS: csak HeaderTypeDeclarationContext
    raw_block_hits = [i for i, s in enumerate(scores_block.tolist()) if s > THRESHOLD_BLOCK]
    block_hits = [i for i in raw_block_hits
                  if graph.nodes[node_ids[i]].get("class_") == "HeaderTypeDeclarationContext"]

    print_predictions("📦 HeaderBlockClassifier (HeaderTypeDeclarationContext)", block_hits, graph, THRESHOLD_BLOCK)

    # célzott debug a konkrét headerre (3273)
    _print_node_score(graph, 3273, scores_block)
    if not block_hits:
        top = _topk_debug(scores_block, graph, k=10)
        if top:
            print("ℹ️ HeaderBlock top-10 jelölt (score, nodeId, class, value):")
            for r, s, nid, c, v in top:
                print(f"   {r:2d}. {s:0.4f} | id={nid} | class={c} | value={v}")

    # --- HeaderFieldClassifier ---
    field_model.eval()
    emb_field = field_model.gnn(pyg_field.x, pyg_field.edge_index)
    logits_field = field_model.head(emb_field).squeeze(-1)
    scores_field = torch.sigmoid(logits_field)

    # Kandidáta halmaz: a Block által pozitívnak ítélt header-gyökerek leszármazottai
    pred_headers = [node_ids[i] for i in block_hits]
    cand = set()
    for h in pred_headers:
        q, vis = [h], set([h])
        while q:
            cur = q.pop(0)
            cand.add(cur)
            for nxt in graph.successors(cur):
                if nxt not in vis:
                    vis.add(nxt);
                    q.append(nxt)

    if cand:
        cand_idx = [i for i, nid in enumerate(node_ids) if nid in cand]
    else:
        cand_idx = list(range(len(node_ids)))  # fallback: teljes gráf

    # küszöbölés + OSZTÁLYSZŰRÉS: csak StructFieldContext a jelölteken belül
    field_hits = [i for i in cand_idx
                  if scores_field[i] > THRESHOLD_FIELD
                  and graph.nodes[node_ids[i]].get("class_") == "StructFieldContext"]

    print_predictions("📐 HeaderFieldClassifier (StructFieldContext-ek)", field_hits, graph, THRESHOLD_FIELD)

    # célzott debug a konkrét fieldre (3283)
    _print_node_score(graph, 3283, scores_field)

    if not field_hits:
        # top-10 a jelölteken belül, CSAK StructFieldContext
        pairs = [(i, float(scores_field[i])) for i in cand_idx
                 if graph.nodes[node_ids[i]].get("class_") == "StructFieldContext"]
        pairs.sort(key=lambda t: t[1], reverse=True)
        top_pairs = pairs[:10]
        if top_pairs:
            print("ℹ️ HeaderField top-10 StructFieldContext (csak a header-subtree-ben):")
            for rank, (i, sc) in enumerate(top_pairs, 1):
                nid = node_ids[i]
                print(f"   {rank:2d}. {sc:0.4f} | id={nid} | class=StructFieldContext")


def run_predictions(path: str, model_dir: str, device: str):
    graphs = load_graphs_any(path)
    if not graphs:
        print(f"❌ Nem találtam gráfot itt: {path}")
        return
    print(f"📁 {len(graphs)} gráf betöltve a(z) '{path}' forrásból.")
    for name, G in graphs:
        classify_graph(G, name, model_dir, device)


# --------- main ---------
def main():
    device = detect_device()
    print(f"🖥️ Eszköz: {device.upper()}")

    if TRAIN_MODE:
        print("🚀 Tanítási mód bekapcsolva.")
        print(f"  - Adatok: {DATA_DIR}")
        print(f"  - Modellek: {MODEL_DIR}")
        print(f"  - Epochok: {EPOCHS}")
        train_and_save_models(data_dir=DATA_DIR, output_dir=MODEL_DIR, epochs=EPOCHS)
        print("✅ Tanítás kész. Modellek elmentve.")
    else:
        if not os.path.isdir(MODEL_DIR):
            print(f"❌ A modellek mappája nem található: {MODEL_DIR}")
            return
        run_predictions(CLASSIFY_GRAPH_PATH, MODEL_DIR, device=device)
        print("\n✅ Validáció/predikció kész.")


if __name__ == "__main__":
    main()
