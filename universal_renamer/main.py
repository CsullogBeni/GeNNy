import os
import json
from typing import Dict

from universal_terminal_renamer import UniversalTerminalRenamer, load_json

TRAIN = False
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE = "basic_p4_v2_normalized.json"
CKPT = os.path.join(os.path.dirname(__file__), "universal_renamer.ckpt")
FROM_TOKEN = "dstAddr"
TO_TOKEN = "destinationAddress"


def save_checkpoint(model: UniversalTerminalRenamer, path: str):
    obj = {
        "state_gnn": model.gnn.state_dict(),
        "state_cls": model.classifier.state_dict(),
        "classes": model.class_encoder.classes_.tolist(),
        "values": model.value_encoder.classes_.tolist(),
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
    }
    import torch
    torch.save(obj, path)


def load_checkpoint(model: UniversalTerminalRenamer, path: str):
    import torch
    obj = torch.load(path, map_location=model.device)
    model.hidden_dim = obj["hidden_dim"]
    model.dropout = obj["dropout"]
    model.class_encoder.classes_ = np.array(obj["classes"], dtype=object)
    model.value_encoder.classes_ = np.array(obj["values"], dtype=object)
    model._fitted_encoders = True
    model._ensure_model(in_dim=2)  # x = [class_id, value_id]
    model.gnn.load_state_dict(obj["state_gnn"])
    model.classifier.load_state_dict(obj["state_cls"])


if __name__ == "__main__":
    import numpy as np, torch

    model = UniversalTerminalRenamer(hidden_dim=64, dropout=0.1)

    if TRAIN:
        model.fit_from_folder(DATA_DIR, base_filename=BASE, epochs=8)
        save_checkpoint(model, CKPT)
        print(f"Saved to: {CKPT}")

    if not TRAIN:
        if not os.path.exists(CKPT):
            raise SystemExit("No checkpoint found.")
        load_checkpoint(model, CKPT)

        test_graph = load_json(os.path.join(DATA_DIR, BASE))
        picks, new_graph = model.rename_in_graph(test_graph, FROM_TOKEN, TO_TOKEN, return_new_json=True)

        print(f"Chosen nodes: {picks[:20]}{'...' if len(picks) > 20 else ''}")
        out_path = os.path.join(os.path.dirname(__file__), f"preview_{FROM_TOKEN}_to_{TO_TOKEN}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_graph, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {out_path}")
