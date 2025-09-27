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
    """
    Calls the proper training function for EmptyElseDetector with the given parameters.

    Args:
         data_dir: Path to the directory containing `.json` training files with P4 ASTs.
         output_dir: Path to the directory where the model will be saved.
         epochs: Number of training iterations (default: 20).
         hidden_dim: Dimension of hidden layers in both classifiers (default: 64).
    """
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
    """
    Run prediction on a graph or a directory of graphs. Model searches for empty else blocks. Searcher also
    validates the detector.

    Args:
        graph_path (str): Path to a JSON file containing a graph or a directory of graphs.
        device (str, optional): Device to run the model on, e.g. `"cpu"` or `"cuda"`. Default is `"cpu"`.
    """
    import json
    from empty_else_searcher import load_ast, find_empty_else_blocks
    detector = EmptyElseDetector(device=device)
    detector.load_model(os.path.dirname(__file__))

    if os.path.isdir(graph_path):
        files = [os.path.join(graph_path, f) for f in os.listdir(graph_path) if f.endswith(".json")]
    elif os.path.isfile(graph_path) and graph_path.endswith(".json"):
        files = [graph_path]
    else:
        print(f"There is no JSON file: {graph_path}")
        return

    for fp in sorted(files):
        print(f"\n=== {os.path.basename(fp)} ===")

        try:
            ast = load_ast(Path(fp))
            searcher_empties = find_empty_else_blocks(ast)
            print(f"[Searcher] empty else block(s): {len(searcher_empties)}")
        except Exception as e:
            print(f"[Searcher] error: {e}")

        with torch.no_grad():
            node_emb = detector.encode_graph(fp)
            pred_idx = detector.predict_subgraph(fp, node_emb)

        with open(fp, "r", encoding="utf-8") as f:
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


def main() -> None:
    """
    Program entry point.

    Behavior depends on `TRAIN_MODE`:
      - If True, trains both models on data from `DATA_DIR` for `EPOCHS` epochs.
      - If False, loads models and classifies graphs found at
        `CLASSIFY_GRAPH_PATH`.
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
