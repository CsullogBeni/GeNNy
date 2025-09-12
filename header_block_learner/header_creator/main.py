"""
main.py – Runner for HeaderCompletionModel (training / validation-time batch inference)

Usage
-----
- TRAIN_MODE = True  → train on (full, reduced) pairs found in PAIRS_DIR and save to MODEL_DIR.
- TRAIN_MODE = False → run header completion on all *.json files under VALIDATION_DIR.

Configuration
-------------
1) Built-in (below):
   - DEFAULT_ADDITIONS: base additions applied to every file.
   - PER_FILE_ADDITIONS: per-file/per-glob overrides/additions.
   - VALIDATION_OUT_DIR, RECURSIVE: output directory & directory traversal.
   - OUTPUT_SUFFIX, OVERWRITE: output naming/overwrite rules (used only if no output_dir is set).

2) External JSON (optional): VALIDATION_CONFIG_PATH (validation/config.json)
   Schema:
   {
     "global_additions": { "my_header": [{"type": "macAddr_t", "name": "dst"}], "3273": [{"type": "u16", "name": "len"}] },
     "per_file": {
       "file_a_reduced.json": {"my_header": [{"type": "macAddr_t", "name": "src"}]},
       "*_reduced.json": {"ipv4_t": [{"type": "bit<8>", "name": "ttl"}]},
       "subdir/only_this.json": {"ethernet_t": [{"type": "macAddr_t", "name": "destinationAddress"}]}
     },
     "output_dir": "validation_out",        // OPTIONAL: takes precedence over overwrite/suffix
     "output_suffix": ".completed.json",    // used only if output_dir is not set
     "overwrite": false,                    // used only if output_dir is not set
     "recursive": true                      // OPTIONAL: recurse into subdirectories
   }
   Note: numeric keys (node_id) must be provided as strings in JSON; the script converts them to ints.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Union, Optional
from fnmatch import fnmatch
from pathlib import Path
import torch

from header_completion_model import HeaderCompletionModel

# CONFIG
TRAIN_MODE: bool = False  # True -> train; False -> validate/infer
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Train
PAIRS_DIR: str = "data"
EPOCHS: int = 20
MODEL_DIR: str = "models"

# Validation / Inference
VALIDATION_DIR: str = "validation"
VALIDATION_OUT_DIR: Optional[str] = "validation_out"  # if set, write outputs here mirroring structure
RECURSIVE: bool = True  # traverse subdirectories under VALIDATION_DIR

# If VALIDATION_OUT_DIR is None, we fall back to suffix/overwrite behavior:
OUTPUT_SUFFIX: str = ".completed.json"
OVERWRITE: bool = False

VALIDATION_CONFIG_PATH: str = os.path.join(VALIDATION_DIR, "config.json")

DEFAULT_ADDITIONS: Dict[Union[int, str], List[Dict[str, Optional[str]]]] = {
    # "my_header": [{"type": "macAddr_t", "name": "destinationAddress"}],
}

PER_FILE_ADDITIONS: Dict[str, Dict[Union[int, str], List[Dict[str, Optional[str]]]]] = \
    {
        # "exact_name.json": {"my_header": [{"type": "macAddr_t", "name": "dstAddr"}]},
        # "subdir/*.json": {"ipv4_t": [{"type": "bit<8>", "name": "ttl"}]},
        "*basic_p4.json": {
            "ethernet_t": [
                {"type": "macAddr_t", "name": "destinationAddress"},
                {"type": "macAddr_t", "name": "sourceAddress"}
            ],
            "ipv4_t": [
                {"type": "ip4Addr_t", "name": "destinationAddress"},
                {"type": "ip4Addr_t", "name": "sourceAddress"}
            ]
        },
        "*basic_p4_with_new_header_validation_0.json": {"my_header": [
            {"type": "macAddr_t", "name": "destinationAddress"},
            {"type": "macAddr_t", "name": "sourceAddress"}
        ]},
        "*basic_p4_2.json": {
            "ethernet_t": [
                {"type": "macAddr_t", "name": "destinationAddress"},
                {"type": "macAddr_t", "name": "sourceAddress"}
            ],
            "ipv4_t": [
                {"type": "ip4Addr_t", "name": "destinationAddress"},
                {"type": "ip4Addr_t", "name": "sourceAddress"}
            ]
        },
        "*basic_p4_4.json": {
            "ethernet_t": [
                {"type": "macAddr_t", "name": "destinationAddress"},
                {"type": "macAddr_t", "name": "sourceAddress"}
            ],
            "ipv4_t": [
                {"type": "ip4Addr_t", "name": "destinationAddress"},
                {"type": "ip4Addr_t", "name": "sourceAddress"}
            ]
        },
        "*basic_p4_8.json": {
            "ethernet_t": [
                {"type": "macAddr_t", "name": "destinationAddress"},
                {"type": "macAddr_t", "name": "sourceAddress"}
            ],
            "ipv4_t": [
                {"type": "ip4Addr_t", "name": "destinationAddress"},
                {"type": "ip4Addr_t", "name": "sourceAddress"}
            ]
        },
        "*basic_p4_16.json": {
            "ethernet_t": [
                {"type": "macAddr_t", "name": "destinationAddress"},
                {"type": "macAddr_t", "name": "sourceAddress"}
            ],
            "ipv4_t": [
                {"type": "ip4Addr_t", "name": "destinationAddress"},
                {"type": "ip4Addr_t", "name": "sourceAddress"}
            ]
        },
        "*ex1.json": {
            "ethernet_t": [
                {"type": "macAddr_t", "name": "destinationAddress"},
                {"type": "macAddr_t", "name": "sourceAddress"}
            ],
            "ipv4_t": [
                {"type": "ip4Addr_t", "name": "destinationAddress"},
                {"type": "ip4Addr_t", "name": "sourceAddress"}
            ]
        },
        "*fabric.json": {
            "ethernet_t": [
                {"type": "macAddr_t", "name": "destinationAddress"},
                {"type": "macAddr_t", "name": "sourceAddress"}
            ],
            "ipv4_t": [
                {"type": "ip4Addr_t", "name": "destinationAddress"},
                {"type": "ip4Addr_t", "name": "sourceAddress"}
            ]
        },
    }


def _normalize_additions_keys(additions: Dict[Union[int, str], List[Dict[str, Optional[str]]]]) -> Dict[
    Union[int, str], List[Dict[str, Optional[str]]]]:
    out: Dict[Union[int, str], List[Dict[str, Optional[str]]]] = {}
    for k, v in (additions or {}).items():
        kk: Union[int, str]
        if isinstance(k, int):
            kk = k
        else:
            ks = str(k).strip()
            kk = int(ks) if ks.isdigit() else ks
        out[kk] = list(v or [])
    return out


def load_external_validation_config(path: str) -> Dict[str, object]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out: Dict[str, object] = {}
    if "global_additions" in cfg:
        out["global_additions"] = _normalize_additions_keys(cfg.get("global_additions") or {})
    if "per_file" in cfg:
        per_file: Dict[str, Dict[Union[int, str], List[Dict[str, Optional[str]]]]] = {}
        for patt, adds in (cfg.get("per_file") or {}).items():
            per_file[patt] = _normalize_additions_keys(adds or {})
        out["per_file"] = per_file
    if "output_suffix" in cfg:
        out["output_suffix"] = str(cfg.get("output_suffix"))
    if "overwrite" in cfg:
        out["overwrite"] = bool(cfg.get("overwrite"))
    if "output_dir" in cfg:
        out["output_dir"] = str(cfg.get("output_dir")) if cfg.get("output_dir") else None
    if "recursive" in cfg:
        out["recursive"] = bool(cfg.get("recursive"))
    return out


def merge_additions(base: Dict[Union[int, str], List[Dict[str, Optional[str]]]],
                    extra: Dict[Union[int, str], List[Dict[str, Optional[str]]]]) -> Dict[
    Union[int, str], List[Dict[str, Optional[str]]]]:
    merged: Dict[Union[int, str], List[Dict[str, Optional[str]]]] = {}
    for src in (base or {}), (extra or {}):
        for k, v in src.items():
            merged.setdefault(k, []).extend(v or [])
    return merged


def additions_for_file(rel_path: str,
                       global_adds: Dict[Union[int, str], List[Dict[str, Optional[str]]]],
                       per_file: Dict[str, Dict[Union[int, str], List[Dict[str, Optional[str]]]]]) -> Dict[
    Union[int, str], List[Dict[str, Optional[str]]]]:
    adds = dict(global_adds or {})
    rel_norm = rel_path.replace("\\", "/")
    for patt, extra in (per_file or {}).items():
        if fnmatch(rel_norm, patt):
            adds = merge_additions(adds, extra)
    return _normalize_additions_keys(adds)


def iter_validation_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def complete_directory(model: HeaderCompletionModel,
                       validation_dir: str,
                       output_dir: Optional[str],
                       output_suffix: str,
                       overwrite: bool,
                       recursive: bool,
                       global_adds: Dict[Union[int, str], List[Dict[str, Optional[str]]]],
                       per_file_adds: Dict[str, Dict[Union[int, str], List[Dict[str, Optional[str]]]]]) -> None:
    root = Path(validation_dir)
    if not root.exists() or not root.is_dir():
        print(f"[Validation] Directory not found: {validation_dir}")
        return

    files = iter_validation_files(root, recursive=recursive)
    if not files:
        print(f"[Validation] No .json files in: {validation_dir}")
        return

    if output_dir:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[Validation] Output dir: {out_root.resolve()} (mirroring structure)")

    print(f"[Validation] Found {len(files)} json file(s) in {validation_dir} (recursive={recursive})")

    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        adds = additions_for_file(rel, global_adds, per_file_adds)
        if not adds:
            print(f"  - {rel}: no additions configured → SKIP")
            continue

        if output_dir:
            out_path = Path(output_dir) / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = p if overwrite else p.with_name(p.stem + output_suffix)

        try:
            written = model.complete_graph(str(p), adds, output_path=str(out_path))
            print(f"  - {rel} → OK  (written: {written})")
        except Exception as e:
            print(f"  - {rel} → ERROR: {e}")


def run() -> None:
    device = DEVICE
    model = HeaderCompletionModel(device=device)

    if TRAIN_MODE:
        print("[HeaderCompletionModel] Training on pairs…")
        model.fit_on_pairs_dir(PAIRS_DIR, epochs=EPOCHS)
        model.save_model(MODEL_DIR)
        return

    print("[HeaderCompletionModel] Completing graphs in validation dir…")
    if not os.path.isdir(MODEL_DIR):
        print(f"Model directory not found: {MODEL_DIR}")
        return
    model.load_model(MODEL_DIR)

    ext = load_external_validation_config(VALIDATION_CONFIG_PATH)

    global_adds = _normalize_additions_keys(DEFAULT_ADDITIONS)
    per_file_adds = dict(PER_FILE_ADDITIONS)

    output_dir = VALIDATION_OUT_DIR
    recursive = RECURSIVE

    if ext:
        if ext.get("global_additions"):
            global_adds = merge_additions(global_adds, ext["global_additions"])
        if ext.get("per_file"):
            for patt, adds in ext["per_file"].items():
                if patt in per_file_adds:
                    per_file_adds[patt] = merge_additions(per_file_adds[patt], adds)
                else:
                    per_file_adds[patt] = adds
        if ext.get("output_dir") is not None:
            output_dir = ext["output_dir"]
        if ext.get("output_suffix"):
            global OUTPUT_SUFFIX
            OUTPUT_SUFFIX = ext["output_suffix"]
        if "overwrite" in ext:
            global OVERWRITE
            OVERWRITE = bool(ext["overwrite"])
        if "recursive" in ext:
            recursive = bool(ext["recursive"])

    complete_directory(
        model=model,
        validation_dir=VALIDATION_DIR,
        output_dir=output_dir,
        output_suffix=OUTPUT_SUFFIX,
        overwrite=OVERWRITE,
        recursive=recursive,
        global_adds=global_adds,
        per_file_adds=per_file_adds,
    )


if __name__ == "__main__":
    run()
