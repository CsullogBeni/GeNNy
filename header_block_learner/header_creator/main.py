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
   - OUTPUT_SUFFIX, OVERWRITE: output naming/overwrite rules.

2) External JSON (optional): VALIDATION_CONFIG_PATH
   Schema:
   {
     "global_additions": { "my_header": [{"type": "macAddr_t", "name": "dst"}], "3273": [{"type": "u16", "name": "len"}] },
     "per_file": {
       "file_a_reduced.json": {"my_header": [{"type": "macAddr_t", "name": "src"}]},
       "*_reduced.json": {"ipv4_t": [{"type": "bit<8>", "name": "ttl"}]}
     },
     "output_suffix": ".completed.json",
     "overwrite": false
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
OUTPUT_SUFFIX: str = ".completed.json"
OVERWRITE: bool = False

VALIDATION_CONFIG_PATH: str = os.path.join(VALIDATION_DIR, "config.json")

DEFAULT_ADDITIONS: Dict[Union[int, str], List[Dict[str, Optional[str]]]] = {
    # "my_header": [{"type": "macAddr_t", "name": "destinationAddress"}],
}

PER_FILE_ADDITIONS: Dict[str, Dict[Union[int, str], List[Dict[str, Optional[str]]]]] = {
    # "*_reduced.json": {"my_header": [{"type": "macAddr_t", "name": "dstAddr"}]},
    # "example_reduced.json": { 3273: [{"type": "u16", "name": "len"}] },
    "*.json": {"my_header": [{"type": "macAddr_t", "name": "destinationAddress"}]},
}


def _normalize_additions_keys(additions: Dict[Union[int, str], List[Dict[str, Optional[str]]]]) -> Dict[
    Union[int, str], List[Dict[str, Optional[str]]]]:
    """
    Normalize the keys of an additions mapping so numeric strings become ints.

    This allows JSON configurations to specify header node ids as strings while
    the in-memory representation uses integers where appropriate.

    Args:
        additions: Mapping from header identifier (int node id or header name)
            to a list of field dicts (each with optional "type" and "name").

    Returns:
        A new mapping with keys converted: numeric strings → int, others → str.
    """
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
    """
    Load optional external validation configuration from JSON.

    The file may define global additions, per-file additions, output suffix,
    and overwrite behavior. Numeric header ids must be strings in JSON and
    are normalized to ints.

    Args:
        path: Filesystem path to the JSON configuration.

    Returns:
        A dictionary with any of the keys:
          - "global_additions": Dict[Union[int, str], List[Dict[str, Optional[str]]]]
          - "per_file": Dict[str, Dict[Union[int, str], List[Dict[str, Optional[str]]]]]
          - "output_suffix": str
          - "overwrite": bool
        Missing file results in an empty dict.
    """
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
    return out


def merge_additions(base: Dict[Union[int, str], List[Dict[str, Optional[str]]]],
                    extra: Dict[Union[int, str], List[Dict[str, Optional[str]]]]) -> Dict[
    Union[int, str], List[Dict[str, Optional[str]]]]:
    """
    Merge two additions dictionaries by concatenating field lists per key.

    Later entries do not replace earlier ones; they are appended, allowing
    the same header to receive multiple field specs.

    Args:
        base: The starting additions mapping.
        extra: The additions to merge into ``base``.

    Returns:
        A new mapping with concatenated lists per header key.
    """
    merged: Dict[Union[int, str], List[Dict[str, Optional[str]]]] = {}
    for src in (base or {}), (extra or {}):
        for k, v in src.items():
            merged.setdefault(k, []).extend(v or [])
    return merged


def additions_for_file(rel_path: str,
                       global_adds: Dict[Union[int, str], List[Dict[str, Optional[str]]]],
                       per_file: Dict[str, Dict[Union[int, str], List[Dict[str, Optional[str]]]]]) -> Dict[
    Union[int, str], List[Dict[str, Optional[str]]]]:
    """
    Compute the effective additions for a given file, combining global and per-file configs.

    Per-file entries are matched using Unix shell-style glob patterns. Multiple
    matching patterns are merged cumulatively.

    Args:
        rel_path: The file name (or path relative to the validation root).
        global_adds: Additions applied to every file.
        per_file: Mapping of glob pattern → additions for matching files.

    Returns:
        A normalized additions mapping for the specific file.
    """
    adds = dict(global_adds or {})
    rel_norm = rel_path.replace("\\", "/")
    for patt, extra in (per_file or {}).items():
        if fnmatch(rel_norm, patt):
            adds = merge_additions(adds, extra)
    return _normalize_additions_keys(adds)


def complete_directory(model: HeaderCompletionModel,
                       validation_dir: str,
                       output_suffix: str,
                       overwrite: bool,
                       global_adds: Dict[Union[int, str], List[Dict[str, Optional[str]]]],
                       per_file_adds: Dict[str, Dict[Union[int, str], List[Dict[str, Optional[str]]]]]) -> None:
    """
    Run header completion over all JSON graphs in a directory.

    For each ``*.json`` file found, the function determines the effective
    additions (global + per-file), invokes :meth:`HeaderCompletionModel.complete_graph`,
    and writes the output either in-place (if ``overwrite``) or to a sibling
    file with ``output_suffix``.

    Args:
        model: An initialized and loaded ``HeaderCompletionModel`` instance.
        validation_dir: Directory to scan for ``*.json`` files.
        output_suffix: Suffix appended to output file names when not overwriting.
        overwrite: If True, write results to the same file path.
        global_adds: Base additions applied to every file.
        per_file_adds: Per-glob additions that may override/extend the base.
    """
    root = Path(validation_dir)
    if not root.exists() or not root.is_dir():
        print(f"[Validation] Directory not found: {validation_dir}")
        return

    files = sorted([p for p in root.glob("*.json") if p.is_file()])
    if not files:
        print(f"[Validation] No .json files in: {validation_dir}")
        return

    print(f"[Validation] Found {len(files)} json file(s) in {validation_dir}")

    for p in files:
        rel = p.name
        adds = additions_for_file(rel, global_adds, per_file_adds)
        if not adds:
            print(f"  - {rel}: no additions configured → SKIP")
            continue
        out_path = p if overwrite else p.with_name(p.stem + output_suffix)
        try:
            written = model.complete_graph(str(p), adds, output_path=str(out_path))
            print(f"  - {rel} → OK  (written: {written})")
        except Exception as e:
            print(f"  - {rel} → ERROR: {e}")


def run() -> None:
    """
    Entry point for training or validation-time completion.

    Behavior is controlled by module-level configuration:
      * If ``TRAIN_MODE`` is True:
          - Train on pairs in ``PAIRS_DIR`` for ``EPOCHS`` epochs,
            then save to ``MODEL_DIR``.
      * If ``TRAIN_MODE`` is False:
          - Load a model from ``MODEL_DIR``.
          - Read optional external config from ``VALIDATION_CONFIG_PATH``.
          - Merge built-in and external additions.
          - Run completion over all ``*.json`` in ``VALIDATION_DIR``.

    Returns:
        None. Prints progress and results to stdout.
    """
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

    if ext:
        if ext.get("global_additions"):
            global_adds = merge_additions(global_adds, ext["global_additions"])
        if ext.get("per_file"):
            for patt, adds in ext["per_file"].items():
                if patt in per_file_adds:
                    per_file_adds[patt] = merge_additions(per_file_adds[patt], adds)
                else:
                    per_file_adds[patt] = adds
        if ext.get("output_suffix"):
            global OUTPUT_SUFFIX
            OUTPUT_SUFFIX = ext["output_suffix"]
        if "overwrite" in ext:
            global OVERWRITE
            OVERWRITE = bool(ext["overwrite"])

    complete_directory(
        model=model,
        validation_dir=VALIDATION_DIR,
        output_suffix=OUTPUT_SUFFIX,
        overwrite=OVERWRITE,
        global_adds=global_adds,
        per_file_adds=per_file_adds,
    )


if __name__ == "__main__":
    run()
