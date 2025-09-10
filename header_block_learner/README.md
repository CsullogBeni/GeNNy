# Header Block Learner

Neural models that learn over **AST graphs generated from P4 programs**. The repository contains two related projects:

- **`header_classifier/`** – Graph neural network (GNN) classifiers that
  1) predict *header block roots/contexts* in the AST, and
  2) predict `StructFieldContext` nodes (*fields*) inside detected header subgraphs.
- **`header_creator/`** – A header **completion** model that can *extend* an existing header subgraph by proposing additional fields (graph augmentation / completion).

Both projects operate on the same JSON-serialized AST graph format (see _Data format_).

---

> Note: the repository depends on some sibling/local packages that are **not** included in this zip:
> - `graph_learner.abstract_graph_learner` (base learner interface / utilities)
> - `graph.graph_builder` (load/save and manipulate AST graphs)
> - `pretty_printer.pretty_printer` (print AST back to P4-like text)
>
> Make sure these packages are importable (e.g. install from their repos, or add them to `PYTHONPATH`).

---

## Data format

All models expect AST graphs serialized to JSON with the following structure (NetworkX-style):

```json
{
  "nodes": [{"id": 12, "label": "StructFieldContext", "line": 73, ...}, ...],
  "edges": [{"source": 12, "target": 13}, ...],
  "graph": {"name": "basic_p4_with_new_header_0", ...}
}
```

- **Nodes** carry at least: `id` (int), `label` (nonterminal/role), optional `line` (source line).
- **Edges** represent directed AST relations.
- Sample graphs live under:
  - `header_classifier/data/`
  - `header_creator/validation/` and `header_creator/data/`

---

## Usage

### 1) Header classifiers (`header_classifier/`)

Two GNNs are trained/used together:
- **HeaderBlockClassifier** – predicts header block *root* nodes (pattern leading to `HeaderTypeDeclarationContext`).
- **HeaderFieldClassifier** – predicts `StructFieldContext` nodes (fields) inside a detected header subtree.

The entry-point script supports **train** and **inference/validation** modes controlled by a constant.

Open `header_block_learner/header_classifier/main.py` and adjust the configuration block near the top:

- `TRAIN_MODE`: `True` to train, `False` to run predictions.
- `DATA_DIR`: folder with JSON graphs for training (e.g. `header_classifier/data`).
- `MODEL_DIR`: where to save/load model checkpoints (e.g. `header_classifier/models`).
- `CLASSIFY_GRAPH_PATH`: file or directory with graphs to classify when `TRAIN_MODE=False`.
- `EPOCHS`: number of training epochs.

Run:
```bash
# Train both classifiers and save checkpoints
python -m header_block_learner.header_classifier.main

# After training, flip TRAIN_MODE=False and run predictions:
python -m header_block_learner.header_classifier.main
```

During inference the script prints a summary:
- number of files processed,
- how many contained header roots,
- total predicted header roots and field nodes,
- and a pretty-printed preview per file (requires `pretty_printer`).

**Advanced**: The implementations live in:
- `header_classifier/header_block_classifier.py`
- `header_classifier/header_field_classifier.py`
- `header_classifier/train_headers.py` (training loop / checkpointing)
- `header_classifier/validate_headers.py` (batch classification helpers)


### 2) Header completion (`header_creator/`)

The **HeaderCompletionModel** predicts **additional fields** to extend an existing header subgraph. The runner supports two workflows:

- **Training** from (full header, reduced header) **pairs** found under `header_creator/data/`.
- **Batch completion** over all graphs in `header_creator/validation/` using a saved model.

Configuration is embedded at the top of `header_creator/main.py`:
- `TRAIN_MODE`: `True` to train, `False` to complete headers in `VALIDATION_DIR`.
- `PAIRS_DIR`: directory with training pairs (full vs. reduced headers).
- `MODEL_DIR`: where checkpoints are stored (defaults to `header_creator/models/` with a sample `header_completion_model.pt`).
- `VALIDATION_DIR`: directory with `.json` graphs to complete.
- `VALIDATION_CONFIG_PATH` (optional): external JSON with **per-file additions** / overrides.
- `DEFAULT_ADDITIONS`, `PER_FILE_ADDITIONS`: built-in additions.
- `OUTPUT_SUFFIX`, `OVERWRITE`: output naming and overwrite policy.

Run:
```bash
# Train / save
python -m header_block_learner.header_creator.main

# Complete headers in validation set (flip TRAIN_MODE=False first)
python -m header_block_learner.header_creator.main
```

Utilities:
- `header_creator/dataset_creator.py` – builds reduced subgraphs from full graphs.
- `header_creator/validation_checker.py` – pretty-prints validation graphs back to P4 for quick inspection.

> The creator relies on `graph.graph_builder.GraphBuilder` and `pretty_printer.PrettyPrinter` to read/write graphs and render P4. Ensure those packages are installed/importable.


---

## Tips & conventions

- All node ids are **integers**; config files may use string ids but are normalized to ints internally.
- Graphs are expected to be **acyclic** and to have consistent parent→child direction as produced by the AST builder used in your pipeline.
- Checkpoints are standard `torch.save` bundles that include the model and optimizer states plus encoders where applicable (see `*_classifier.load(...)`).

---

## Troubleshooting

- `ModuleNotFoundError: graph_learner ...`  
  → Install the missing package(s) or add their folders to `PYTHONPATH`.

- `torch_geometric` installation errors  
  → Ensure your `torch` version matches the prebuilt wheels. See the PyG install docs.

- Pretty-printer output empty/garbled  
  → Verify the AST graph has `label`/`line` attributes and edge directions expected by your `PrettyPrinter` implementation.
