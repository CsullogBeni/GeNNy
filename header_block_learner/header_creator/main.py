# -*- coding: utf-8 -*-
"""
main.py – HeaderCompletionModel futtató (train / validation batch inferencia)

Használat
---------
- TRAIN_MODE = True  → páros tanítás a PAIRS_DIR mappán, mentés a MODEL_DIR-be.
- TRAIN_MODE = False → kiegészítés futtatása a VALIDATION_DIR-ben lévő összes *.json fájlra.

Konfiguráció
------------
1) Beépített (alább):
   - DEFAULT_ADDITIONS: minden fájlra alkalmazandó alap kiegészítés.
   - PER_FILE_ADDITIONS: fájlonként/glob minta szerint felülírás/hozzáadás.
   - OUTPUT_SUFFIX, OVERWRITE: kimeneti név/szabályok.

2) Külső JSON (opcionális): VALIDATION_CONFIG_PATH
   Sémája:
   {
     "global_additions": { "my_header": [{"type": "macAddr_t", "name": "dst"}], "3273": [{"type": "u16", "name": "len"}] },
     "per_file": {
       "file_a_reduced.json": {"my_header": [{"type": "macAddr_t", "name": "src"}]},
       "*_reduced.json": {"ipv4_t": [{"type": "bit<8>", "name": "ttl"}]}
     },
     "output_suffix": ".completed.json",
     "overwrite": false
   }
   Megjegyzés: a numerikus kulcsokat (node_id) sztringként kell megadni a JSON-ban, a script intté konvertálja.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
from fnmatch import fnmatch
from pathlib import Path

import torch

from header_completion_model import HeaderCompletionModel

# =====================
# ALAP KONFIG
# =====================
TRAIN_MODE: bool = False  # True → train; False → validate/infer
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Train
PAIRS_DIR: str = "data"  # páros dataset (teljes & *_reduced.json)
EPOCHS: int = 20
MODEL_DIR: str = "models"  # ide ment/innen tölt a modell

# Validation / Inference
VALIDATION_DIR: str = "validation"  # ide teszed a tesztelendő *.json fájlokat
OUTPUT_SUFFIX: str = ".completed.json"
OVERWRITE: bool = False

# Opcionális külső konfig (ha létezik, felülírja/egyesíti az alábbiakat)
VALIDATION_CONFIG_PATH: str = os.path.join(VALIDATION_DIR, "config.json")

# Alapértelmezett kiegészítések (minden fájlra)
DEFAULT_ADDITIONS: Dict[Union[int, str], List[Dict[str, Optional[str]]]] = {
    # "my_header": [{"type": "macAddr_t", "name": "destinationAddress"}],
}

# Fájl-specifikus/glob-specifikus kiegészítések (hozzáadódnak az alaphoz)
# Kulcs: fájlnév vagy glob minta a VALIDATION_DIR alatti relatív útvonalra.
PER_FILE_ADDITIONS: Dict[str, Dict[Union[int, str], List[Dict[str, Optional[str]]]]] = {
    # "*_reduced.json": {"my_header": [{"type": "macAddr_t", "name": "dstAddr"}]},
    # "example_reduced.json": { 3273: [{"type": "u16", "name": "len"}] },
    "*.json": {"my_header": [{"type": "macAddr_t", "name": "destinationAddress"}]},
}


# =====================
# SEGÉDFÜGGVÉNYEK
# =====================

def _normalize_additions_keys(additions: Dict[Union[int, str], List[Dict[str, Optional[str]]]]) -> Dict[
    Union[int, str], List[Dict[str, Optional[str]]]]:
    """JSON-ból érkező kulcsok str-k lehetnek; a pusztán számjegyekből állókat int-té konvertáljuk (node_id)."""
    out: Dict[Union[int, str], List[Dict[str, Optional[str]]]] = {}
    for k, v in (additions or {}).items():
        kk: Union[int, str]
        if isinstance(k, int):
            kk = k
        else:
            # ha egész számnak tűnik → int
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
        # Minden minta alá normalizáljuk a kulcsokat
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
    # Alkalmazzuk a pattern-öket
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
        rel = p.name  # csak fájlnév (VALIDATION_DIR gyökerében dolgozunk)
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


# =====================
# FŐ FUTTATÓ
# =====================

def run() -> None:
    device = DEVICE
    model = HeaderCompletionModel(device=device)

    if TRAIN_MODE:
        print("[HeaderCompletionModel] Training on pairs…")
        model.fit_on_pairs_dir(PAIRS_DIR, epochs=EPOCHS)
        model.save_model(MODEL_DIR)
        return

    # Inference / Validation batch
    print("[HeaderCompletionModel] Completing graphs in validation dir…")
    if not os.path.isdir(MODEL_DIR):
        print(f"Model directory not found: {MODEL_DIR}")
        return
    model.load_model(MODEL_DIR)

    # Külső konfig betöltése (ha van)
    ext = load_external_validation_config(VALIDATION_CONFIG_PATH)

    global_adds = _normalize_additions_keys(DEFAULT_ADDITIONS)
    per_file_adds = dict(PER_FILE_ADDITIONS)

    if ext:
        if ext.get("global_additions"):
            global_adds = merge_additions(global_adds, ext["global_additions"])  # unió
        if ext.get("per_file"):
            # minta → additions
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
