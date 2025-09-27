import os
from pathlib import Path
from typing import List
import torch

from empty_else_detector import EmptyElseDetector

TRAIN_MODE = False
CLASSIFY_GRAPH_PATH = "test_files"  # can be a directory or a single file
DATA_DIR = "data"
EPOCHS = 20


def detect_device() -> str:
    """
    Detect the best available computation device.

    Returns:
        str: "cuda" if a CUDA-capable GPU is available via PyTorch,
             otherwise "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_and_save_models(data_dir: str, output_dir: str,
                          epochs: int = 20, hidden_dim: int = 64):
    json_files: List[str] = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")
    ]

    block_model = EmptyElseDetector(hidden_dim=hidden_dim)
    print("Training: EmptyElseDetector")
    block_model.fit(json_files, epochs=epochs)
    os.makedirs(output_dir, exist_ok=True)
    block_model.save_model(output_dir)
    print("EmptyElseDetector saved")


def run_predictions(graph_path: str, device: str = "cpu"):
    import json
    from empty_else_searcher import load_ast, find_empty_else_blocks  # baseline kereső
    detector = EmptyElseDetector(device=device)
    # A modellek és encoderek a projekt gyökeréből töltődnek (ahová mentetted őket)
    detector.load_model(os.path.dirname(__file__))

    if os.path.isdir(graph_path):
        files = [os.path.join(graph_path, f) for f in os.listdir(graph_path) if f.endswith(".json")]
    elif os.path.isfile(graph_path) and graph_path.endswith(".json"):
        files = [graph_path]
    else:
        print(f"Nincs feldolgozható JSON itt: {graph_path}")
        return

    for fp in sorted(files):
        print(f"\n=== {os.path.basename(fp)} ===")

        # 1) baseline szabályalapú kereső (összehasonlításként)
        try:
            ast = load_ast(Path(fp))
            searcher_empties = find_empty_else_blocks(ast)
            print(f"[Searcher] üres else ág(ak): {len(searcher_empties)} találat")
        except Exception as e:
            print(f"[Searcher] hiba: {e}")
            searcher_empties = []

        # 2) GNN embedding + predikció
        with torch.no_grad():
            node_emb = detector.encode_graph(fp)  # [N, hidden_dim]
            pred_idx = detector.predict_subgraph(fp, node_emb)  # [k index]

        # 3) Predikciók visszamappelése a JSON node-okra, hogy olvasható legyen
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # FONTOS: az AbstractGraphLearner a JSON-ban szereplő node-ok sorrendjét veszi át,
        # ezért itt ugyanebben a sorrendben indexelünk vissza.  :contentReference[oaicite:3]{index=3}
        nodes_in_order = data.get("nodes", [])
        hits = []
        for i in pred_idx:
            if 0 <= i < len(nodes_in_order):
                n = nodes_in_order[i]
                hits.append({
                    "index": i,
                    # A JSON-okban lehet "id" vagy "nodeId" kulcs — próbáljuk mindkettőt.
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
        print(f"  - Epochs: {EPOCHS}")
        train_and_save_models(data_dir=DATA_DIR, output_dir=os.path.dirname(__file__), epochs=EPOCHS)
        print("Training complete. Models saved.")
    else:
        if not os.path.isfile('empty_else_detector_model.pt'):
            print("Previously trained model has not found")
            return
        if not os.path.isfile('empty_else_detector_class_encoder.pkl'):
            print("Previously trained model has not found")
            return
        if not os.path.isfile('empty_else_detector_value_encoder.pkl'):
            print("Previously trained model has not found")
            return
        run_predictions(CLASSIFY_GRAPH_PATH, device=device)
        print("\nValidation/prediction complete.")


if __name__ == "__main__":
    main()
