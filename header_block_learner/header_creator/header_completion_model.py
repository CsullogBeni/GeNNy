from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Any, Union

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from graph_learner.abstract_graph_learner import AbstractGraphLearner


@dataclass
class FieldSpec:
    """
    Lightweight container describing a header field to be added or searched for.

    Attributes:
        type: Optional textual type of the field (e.g., "bit<16>", "mac_addr").
        name: Optional identifier/name of the field (e.g., "etherType").
    """
    type: Optional[str] = None
    name: Optional[str] = None


class HeaderCompletionModel(AbstractGraphLearner):
    """
    Header localization and deterministic AST completion for P4-like headers.

    This model scores header nodes (HeaderTypeDeclarationContext) for a given set
    of field specifications and, optionally, amends the AST by inserting the field
    subtree into the selected header. The GNN encoder and label/value encoders are
    provided by AbstractGraphLearner; this class adds a conditional head for
    header scoring and algorithmic routines for AST mutation.
    """

    def __init__(self, hidden_dim: int = 64, device: str = "cpu",
                 gnn_layers: int = 3, gnn_dropout: float = 0.10):
        super().__init__(hidden_dim=hidden_dim, device=device,
                         gnn_layers=gnn_layers, gnn_dropout=gnn_dropout)
        self.cond_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.10),
            nn.Linear(self.hidden_dim, 1)
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    @staticmethod
    def load_graph_json(path: str) -> nx.DiGraph:
        """
        Load a directed AST graph from a JSON file into a NetworkX DiGraph.

        The JSON is expected to contain a list of ``nodes`` (each with an ``id``
        and arbitrary attributes) and a list of ``edges`` (objects with
        ``source`` and ``target`` ids). Edges where either endpoint is missing
        from the node set are skipped.

        Args:
            path: Filesystem path to the JSON graph.

        Returns:
            A ``nx.DiGraph`` whose nodes carry the attributes from the JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        G = nx.DiGraph()
        for n in data.get("nodes", []):
            G.add_node(n["id"], **n)
        for e in data.get("edges", []):
            s, t = e.get("source"), e.get("target")
            if s in G and t in G:
                G.add_edge(s, t)
        return G

    def _load_graph_from_json(self, path: str) -> nx.DiGraph:
        """
        Compatibility wrapper around :meth:`load_graph_json`.

        Some base classes/utilities may define similarly named helpers; this method
        centralizes how graphs are loaded in this implementation.

        Args:
            path: Filesystem path to the JSON graph.

        Returns:
            The loaded ``nx.DiGraph``.
        """
        return self.load_graph_json(path)

    @staticmethod
    def dump_graph_json(G: nx.DiGraph, path: str) -> str:
        """
        Serialize a NetworkX DiGraph back to the expected JSON format.

        Missing common attributes are filled with reasonable defaults to keep
        downstream tooling robust.

        Args:
            G: The graph to serialize.
            path: Output file path.

        Returns:
            The same ``path`` for convenience/chaining.
       """
        nodes = []
        for nid, attrs in G.nodes(data=True):
            a = dict(attrs)
            a.setdefault("id", nid)
            a.setdefault("label", "syn")
            a.setdefault("line", -1)
            a.setdefault("start", -1)
            a.setdefault("end", -1)
            a.setdefault("nodeId", nid)
            a.setdefault("value", None)
            a.setdefault("class_", attrs.get("class_", None))
            nodes.append(a)
        edges = [{"source": s, "target": t} for s, t in G.edges()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def _header_nodes(G: nx.DiGraph) -> List[int]:
        """
        Collect node ids that represent headers in the AST.

        Args:
            G: AST graph.

        Returns:
            List of node ids whose ``class_`` equals ``"HeaderTypeDeclarationContext"``.
        """
        return [n for n in G.nodes if G.nodes[n].get("class_") == "HeaderTypeDeclarationContext"]

    @staticmethod
    def _struct_fields_under(G: nx.DiGraph, header_id: int) -> List[int]:
        """
        Gather all ``StructFieldContext`` nodes in the subtree of a header.

        A simple BFS is used to traverse descendants.

        Args:
            G: AST graph.
            header_id: The node id of the header whose fields are requested.

        Returns:
            A list of node ids representing struct fields under the header.
        """
        out = []
        q, vis = [header_id], set()
        while q:
            cur = q.pop(0)
            if cur in vis:
                continue
            vis.add(cur)
            if G.nodes[cur].get("class_") == "StructFieldContext":
                out.append(cur)
            for nxt in G.successors(cur):
                if nxt not in vis:
                    q.append(nxt)
        return out

    @staticmethod
    def _descendants(G: nx.DiGraph, nid: int) -> Iterable[int]:
        """
        Iterate over all descendants of a node in BFS order.

        Args:
            G: AST graph.
            nid: Start node id.

        Yields:
            Node ids of all descendants (children, grandchildren, ...).
        """
        q, vis = [nid], set()
        while q:
            cur = q.pop(0)
            for nxt in G.successors(cur):
                if nxt not in vis:
                    vis.add(nxt)
                    q.append(nxt)
                    yield nxt

    @staticmethod
    def _first_value_under(G: nx.DiGraph,
                           nid: int,
                           class_path_prefix: Optional[Tuple[str, ...]] = None
                           ) -> Optional[str]:
        """
        Heuristically extract a terminal token value under a subtree.

        The routine scans breadth-first for ``TerminalNodeImpl.value``. If
        ``class_path_prefix`` is provided, values encountered along paths whose
        sequence of ``class_`` names begins with the given prefix are preferred.

        Args:
            G: AST graph.
            nid: Root node id of the subtree to search.
            class_path_prefix: Optional tuple of class names describing a preferred
                prefix along the path (e.g., ``("TypeNameContext",)``).

        Returns:
            The first matching token value according to the preference rule, or
            ``None`` if no terminal value exists under the subtree.
        """
        from collections import deque
        dq = deque([(nid, [])])
        best_any = None
        best_pref = None
        while dq:
            cur, path = dq.popleft()
            c = G.nodes[cur].get("class_")
            if c == "TerminalNodeImpl":
                val = G.nodes[cur].get("value")
                if val is not None:
                    if class_path_prefix is None:
                        return val
                    if best_any is None:
                        best_any = val
                    if len(path) >= len(class_path_prefix) and tuple(
                            path[:len(class_path_prefix)]) == class_path_prefix:
                        if best_pref is None:
                            best_pref = val
            for nxt in G.successors(cur):
                dq.append((nxt, path + [G.nodes[nxt].get("class_")]))
        return best_pref if best_pref is not None else best_any

    @staticmethod
    def _extract_field_tokens(G: nx.DiGraph, field_id: int) -> FieldSpec:
        """
        Extract best-effort type and name tokens from a field subtree.

        The method walks the descendants of a ``StructFieldContext`` and tries to
        recover a type-like token (by prioritizing TypeName/PrefixedType branches)
        and a name-like token (by prioritizing NonTypeName branches).

        Args:
            G: AST graph.
            field_id: Node id of the ``StructFieldContext``.

        Returns:
            A :class:`FieldSpec` with possibly ``None`` components if tokens cannot
            be found.
        """
        type_val = None
        name_val = None
        for d in HeaderCompletionModel._descendants(G, field_id):
            cls = G.nodes[d].get("class_")
            if cls in ("TypeRefContext", "TypeNameContext", "PrefixedTypeContext") and type_val is None:
                type_val = HeaderCompletionModel._first_value_under(G, d, class_path_prefix=("TypeNameContext",))
            if cls in ("NameContext", "NonTypeNameContext") and name_val is None:
                name_val = HeaderCompletionModel._first_value_under(G, d, class_path_prefix=("NonTypeNameContext",))
            if type_val is not None and name_val is not None:
                break
        return FieldSpec(type=type_val, name=name_val)

    @staticmethod
    def _match_pairs_in_dir(pairs_dir: str) -> List[Tuple[str, str]]:
        """
        Find matching (full, reduced) graph JSON pairs in a directory.

        A pair is defined as ``<name>.json`` and ``<name>_reduced.json`` both
        present in ``pairs_dir``.

        Args:
            pairs_dir: Directory to scan.

        Returns:
            A sorted list of tuples ``(full_path, reduced_path)`` for each match.
        """
        files = [f for f in os.listdir(pairs_dir) if f.endswith(".json")]
        reduced = [f for f in files if f.endswith("_reduced.json")]
        out: List[Tuple[str, str]] = []
        for r in sorted(reduced):
            base = r[:-len("_reduced.json")] + ".json"
            full = base
            if full in files:
                out.append((os.path.join(pairs_dir, full), os.path.join(pairs_dir, r)))
        return out

    @staticmethod
    def _missing_fields(fullG: nx.DiGraph, redG: nx.DiGraph) -> List[Tuple[int, int]]:
        """
        Compute which fields present in the full graph are missing in the reduced one.

        Only headers that exist in both graphs are considered. For each common header,
        any ``StructFieldContext`` present in ``fullG`` but absent in ``redG`` is
        reported as missing.

        Args:
            fullG: The full AST graph.
            redG: The reduced AST graph.

        Returns:
            A list of ``(header_id, missing_field_id_in_full)`` tuples.
        """
        miss: List[Tuple[int, int]] = []
        for h in HeaderCompletionModel._header_nodes(fullG):
            if h not in redG:
                continue
            full_fields = set(HeaderCompletionModel._struct_fields_under(fullG, h))
            red_fields = set(HeaderCompletionModel._struct_fields_under(redG, h))
            missing = [fid for fid in full_fields if fid not in red_fields]
            for fid in missing:
                miss.append((h, fid))
        return miss

    def _encode_condition(self, specs: List[FieldSpec]) -> torch.Tensor:
        """
        Encode one or more field specifications into a fixed-size vector.

        Heuristic:
          * Tokenize each spec using its ``type`` and ``name`` (if present).
          * Map tokens through the learned value encoder/autoencoder.
          * Average token embeddings per field, then average across fields.

        Args:
            specs: List of field specifications to encode.

        Returns:
            A tensor of shape ``[hidden_dim]`` on ``self.device``.
        """
        if not specs:
            return torch.zeros(self.hidden_dim, device=self.device)
        vals: List[torch.Tensor] = []
        for sp in specs:
            tokens: List[str] = []
            if sp.type:
                tokens.append(str(sp.type))
            if sp.name:
                tokens.append(str(sp.name))
            if not tokens:
                val_emb = torch.zeros(self.hidden_dim, device=self.device)
            else:
                enc = self._safe_transform(self.value_encoder, tokens, self.VALUE_UNK)
                enc_t = torch.as_tensor(enc, dtype=torch.float32, device=self.device).unsqueeze(1)
                ve = self.value_autoencoder(enc_t)
                val_emb = ve.mean(dim=0)
            vals.append(val_emb)
        return torch.stack(vals, dim=0).mean(dim=0)

    def fit_on_pairs(self, pairs: List[Tuple[str, str]], epochs: int = 20,
                     prune_step: float = 0.05, prune_max_ratio: float = 0.30) -> None:
        """
        Train encoders and the conditional head using (full, reduced) graph pairs.

        For curriculum-like robustness, the reduced graph can be further pruned
        during early epochs by randomly deleting leaves or whole subtrees.

        Args:
            pairs: List of ``(full_json_path, reduced_json_path)`` pairs.
            epochs: Number of training epochs.
            prune_step: Incremental pruning ratio added per epoch.
            prune_max_ratio: Upper bound on pruning ratio.
        """
        all_files: List[str] = []
        for full_p, red_p in pairs:
            all_files.extend([full_p, red_p])
        self.fit_encoders(all_files)

        for epoch in range(epochs):
            ratio = min(prune_step * (epoch + 1), prune_max_ratio)
            total_loss = 0.0
            total_pos = 0
            total_mask = 0

            for full_p, red_p in pairs:
                fullG = self._load_graph_from_json(full_p)
                redG = self._load_graph_from_json(red_p)

                if epoch > 0:
                    if epoch < 3:
                        redG = self._delete_random_leaves(redG, ratio)
                    else:
                        redG = self._delete_random_subtrees(redG, ratio)

                miss = self._missing_fields(fullG, redG)
                if not miss:
                    continue

                cond_specs: List[FieldSpec] = []
                for (_, fid) in miss:
                    cond_specs.append(self._extract_field_tokens(fullG, fid))
                cond_vec = self._encode_condition(cond_specs)

                data = self._graph_to_pyg(redG)
                x = self._encode_node_features(data._raw_node_attrs)
                emb = self.gnn(x, data.edge_index)

                node_ids: List[int] = getattr(data, "_node_ids", list(redG.nodes))
                is_header = torch.tensor([
                    1 if redG.nodes[n].get("class_") == "HeaderTypeDeclarationContext" else 0
                    for n in node_ids
                ], dtype=torch.bool, device=self.device)

                labels = torch.zeros(len(node_ids), dtype=torch.float32, device=self.device)
                for (hid, _fid) in miss:
                    if hid in redG:
                        try:
                            idx = node_ids.index(hid)
                            labels[idx] = 1.0
                        except ValueError:
                            pass

                if is_header.sum().item() == 0:
                    continue

                cond = cond_vec.unsqueeze(0).expand(emb.size(0), -1)
                logits = self.cond_head(torch.cat([emb, cond], dim=1)).squeeze(-1)

                logits_m = logits[is_header]
                labels_m = labels[is_header]

                pos = int(labels_m.sum().item())
                neg = int(labels_m.numel() - pos)
                total_pos += pos
                total_mask += int(labels_m.numel())

                pos_w = torch.tensor([neg / max(pos, 1)], device=self.device, dtype=torch.float32)
                loss = F.binary_cross_entropy_with_logits(logits_m, labels_m, pos_weight=pos_w)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())

            print(f"[epoch {epoch}] loss={total_loss:.4f}  mask_nodes={total_mask}  pos_in_mask={total_pos}")

    def fit_on_pairs_dir(self, pairs_dir: str, epochs: int = 20,
                         prune_step: float = 0.05, prune_max_ratio: float = 0.30) -> None:
        """
        Convenience wrapper to train on all matching pairs in a directory.

        Args:
            pairs_dir: Directory containing ``*.json`` and ``*_reduced.json`` pairs.
            epochs: Number of epochs to train.
            prune_step: Incremental pruning ratio per epoch.
            prune_max_ratio: Max pruning ratio across epochs.
        """
        pairs = self._match_pairs_in_dir(pairs_dir)
        if not pairs:
            print(f"No pairs found in: {pairs_dir}")
            return
        self.fit_on_pairs(pairs, epochs=epochs, prune_step=prune_step, prune_max_ratio=prune_max_ratio)

    @torch.no_grad()
    def score_headers(self, graph_path: str, specs: List[FieldSpec]) -> List[Tuple[int, float]]:
        """
        Score headers in a graph for their compatibility with the given field specs.

        The method encodes the graph with the GNN, conditions the scoring head on
        the aggregated field-spec embedding, and returns sigmoid scores for all
        header nodes.

        Args:
            graph_path: Path to the JSON graph to evaluate.
            specs: List of field specifications to condition on.

        Returns:
            A list of ``(header_node_id, score)`` sorted descending by score.
        """
        G = self._load_graph_from_json(graph_path)
        data = self._graph_to_pyg(G)
        x = self._encode_node_features(data._raw_node_attrs)
        emb = self.gnn(x, data.edge_index)
        node_ids: List[int] = getattr(data, "_node_ids", list(G.nodes))

        cond = self._encode_condition(specs).unsqueeze(0).expand(emb.size(0), -1)
        logits = self.cond_head(torch.cat([emb, cond], dim=1)).squeeze(-1)
        scores = torch.sigmoid(logits)

        out: List[Tuple[int, float]] = []
        for i, nid in enumerate(node_ids):
            if G.nodes[nid].get("class_") == "HeaderTypeDeclarationContext":
                out.append((nid, float(scores[i].item())))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    @staticmethod
    def _ensure_field_list(G: nx.DiGraph, header_id: int) -> int:
        """
        Ensure a ``StructFieldListContext`` exists directly under the header.

        If not found, the method creates a new node, attaches it to the header,
        and returns its id.

        Args:
            G: AST graph.
            header_id: Header node id.

        Returns:
            Node id of the (existing or newly created) field list context.
        """
        for ch in G.successors(header_id):
            if G.nodes[ch].get("class_") == "StructFieldListContext":
                return ch
        next_id = max(int(n) for n in G.nodes if isinstance(n, int)) + 1 if len(G) else 1
        list_id = next_id
        G.add_node(list_id, id=list_id, nodeId=list_id, label="syn", line=-1, start=-1, end=-1,
                   value=None, class_="StructFieldListContext")
        G.add_edge(header_id, list_id)
        return list_id

    @staticmethod
    def _new_id(G: nx.DiGraph, k: int = 1) -> List[int]:
        """
        Allocate one or more fresh integer node ids.

        Args:
            G: AST graph.
            k: Number of consecutive ids to allocate.

        Returns:
            A list of ``k`` fresh ids not currently present in the graph.
        """
        base = max(int(n) for n in G.nodes if isinstance(n, int)) + 1 if len(G) else 1
        return list(range(base, base + k))

    @staticmethod
    def _add_terminal(G: nx.DiGraph, parent: int, value: str) -> int:
        """
        Create a ``TerminalNodeImpl`` with the given value under ``parent``.

        Args:
            G: AST graph.
            parent: Parent node id (edge ``parent → new`` will be added).
            value: Terminal token value to store.

        Returns:
            The id of the newly created terminal node.
        """
        nid = HeaderCompletionModel._new_id(G, 1)[0]
        G.add_node(nid, id=nid, nodeId=nid, label="syn", line=-1, start=-1, end=-1,
                   value=value, class_="TerminalNodeImpl")
        G.add_edge(parent, nid)
        return nid

    @staticmethod
    def _add_node(G: nx.DiGraph, parent: int, cls: str, value: Optional[Any] = None) -> int:
        """
        Create a non-terminal node with class ``cls`` and optional value.

        Args:
            G: AST graph.
            parent: Parent node id; if ``None``, the node is created unattached.
            cls: Value for the node's ``class_`` attribute.
            value: Optional payload for the node's ``value`` attribute.

        Returns:
            The id of the newly created node.
        """
        nid = HeaderCompletionModel._new_id(G, 1)[0]
        G.add_node(nid, id=nid, nodeId=nid, label="syn", line=-1, start=-1, end=-1,
                   value=value, class_=cls)
        if parent is not None:
            G.add_edge(parent, nid)
        return nid

    @staticmethod
    def _build_field_subtree(G: nx.DiGraph, list_id: int, spec: FieldSpec) -> int:
        """
        Materialize a complete ``StructFieldContext`` subtree under a field list.

        The produced structure mirrors typical P4 grammar fragments:
        type branch (``TypeRef → TypeName → PrefixedType → Type_or_id``),
        name branch (``Name → NonTypeName → Type_or_id``), and a trailing ``;``.

        Args:
            G: AST graph.
            list_id: Node id of the parent ``StructFieldListContext``.
            spec: Desired field type/name; missing parts use placeholders.

        Returns:
            The node id of the created ``StructFieldContext``.
        """
        cur = list_id
        visited = set()
        while True:
            if cur in visited:
                # Safety against accidental cycles
                break
            visited.add(cur)

            children = list(G.successors(cur))
            if not children:
                # Found empty tail
                break

            # Partition by class
            field_children = [c for c in children if G.nodes[c].get("class_") == "StructFieldContext"]
            list_children = [c for c in children if G.nodes[c].get("class_") == "StructFieldListContext"]

            if list_children:
                # Follow the next-list (prefer smallest id for determinism)
                cur = sorted(list_children, key=int)[0]
                continue
            elif field_children and not list_children:
                # Legacy/partial node: has a field but no "rest" → create it
                cur = HeaderCompletionModel._add_node(G, cur, "StructFieldListContext")
                break
            else:
                # Unexpected shape → treat as tail
                break

        # At tail: add (StructFieldContext, StructFieldListContext)
        f_id = HeaderCompletionModel._add_node(G, cur, "StructFieldContext")

        # Type-subtree
        tref = HeaderCompletionModel._add_node(G, f_id, "TypeRefContext")
        tname = HeaderCompletionModel._add_node(G, tref, "TypeNameContext")
        tpre = HeaderCompletionModel._add_node(G, tname, "PrefixedTypeContext")
        ttoi = HeaderCompletionModel._add_node(G, tpre, "Type_or_idContext")
        HeaderCompletionModel._add_terminal(G, ttoi, spec.type or "<UNK_TYPE>")

        # Name-subtree
        nname = HeaderCompletionModel._add_node(G, f_id, "NameContext")
        nnon = HeaderCompletionModel._add_node(G, nname, "NonTypeNameContext")
        ntoi = HeaderCompletionModel._add_node(G, nnon, "Type_or_idContext")
        HeaderCompletionModel._add_terminal(G, ntoi, spec.name or "<UNK_NAME>")

        # Terminal ';'
        HeaderCompletionModel._add_terminal(G, f_id, ";")

        # Chain extension: empty list after the field
        HeaderCompletionModel._add_node(G, cur, "StructFieldListContext")

        return f_id

    @staticmethod
    def _find_header_by_name(G: nx.DiGraph, header_name: str) -> List[int]:
        """
        Locate headers whose subtree contains a terminal with the given name.

        Args:
            G: AST graph.
            header_name: The name token to search for.

        Returns:
            List of header node ids matching the name; can be empty or contain
            multiple candidates.
        """
        out: List[int] = []
        for h in HeaderCompletionModel._header_nodes(G):
            found = False
            q = [h]
            vis = set()
            while q and not found:
                cur = q.pop(0)
                if cur in vis:
                    continue
                vis.add(cur)
                if G.nodes[cur].get("class_") == "TerminalNodeImpl" and G.nodes[cur].get("value") == header_name:
                    found = True
                    break
                for nxt in G.successors(cur):
                    if nxt not in vis:
                        q.append(nxt)
            if found:
                out.append(h)
        return out

    def complete_graph(self, graph_path: str,
                       additions: Dict[Union[int, str], List[Dict[str, str]]],
                       output_path: Optional[str] = None) -> str:
        """
        Insert fields into one or more headers and write the completed graph.

        Headers can be addressed either by node id (int) or by header name (str).
        When addressed by name:
          * If no exact-name header exists, the model scores all headers with
            :meth:`score_headers` and picks the best.
          * If multiple name matches exist, the conditional head disambiguates
            by choosing the highest-scoring candidate among the matches.

        Args:
            graph_path: Input graph JSON path.
            additions: Mapping from header id or name to a list of field dicts
                (each dict may contain keys ``"type"`` and ``"name"``).
            output_path: Optional explicit output path. If ``None``, a
                ``.completed.json`` sibling is created.

        Returns:
            The output path written by :meth:`dump_graph_json`.
        """
        G = self._load_graph_from_json(graph_path)

        by_id: Dict[int, List[FieldSpec]] = {}
        by_name: List[Tuple[str, List[FieldSpec]]] = []
        for key, lst in additions.items():
            specs = [FieldSpec(type=d.get("type"), name=d.get("name")) for d in lst]
            if isinstance(key, int):
                by_id.setdefault(key, []).extend(specs)
            else:
                by_name.append((str(key), specs))

        for name, specs in by_name:
            cand = self._find_header_by_name(G, name)
            if not cand:
                scores = self.score_headers(graph_path, specs)
                if not scores:
                    continue
                target = scores[0][0]
                by_id.setdefault(target, []).extend(specs)
            elif len(cand) == 1:
                by_id.setdefault(cand[0], []).extend(specs)
            else:
                specs_vec = self._encode_condition(specs).unsqueeze(0)
                data = self._graph_to_pyg(G)
                x = self._encode_node_features(data._raw_node_attrs)
                emb = self.gnn(x, data.edge_index)
                node_ids: List[int] = getattr(data, "_node_ids", list(G.nodes))
                logits = self.cond_head(torch.cat([emb, specs_vec.expand(emb.size(0), -1)], dim=1)).squeeze(-1)
                scores = torch.sigmoid(logits)
                best = None
                best_sc = -1.0
                for hid in cand:
                    try:
                        i = node_ids.index(hid)
                        sc = float(scores[i])
                        if sc > best_sc:
                            best_sc, best = sc, hid
                    except ValueError:
                        continue
                if best is not None:
                    by_id.setdefault(best, []).extend(specs)

        for hid, specs in by_id.items():
            if hid not in G:
                continue
            list_id = self._ensure_field_list(G, hid)
            for sp in specs:
                self._build_field_subtree(G, list_id, sp)

            self._relocate_header_closing_brace(G, hid)

        out = output_path or (os.path.splitext(graph_path)[0] + ".completed.json")
        return self.dump_graph_json(G, out)

    @staticmethod
    def _relocate_header_closing_brace(G: nx.DiGraph, header_id: int) -> None:
        """
        Remove all closing-brace ('}') TerminalNodeImpl under the given header and
        re-create a single '}' as a **direct child** of the header, appended last.
        This guarantees that:
          - the '}' comes after the StructFieldListContext in successor iteration
          - the new node gets a fresh, larger nodeId (due to _add_terminal).
        """
        direct_closers = [c for c in G.successors(header_id)
                          if G.nodes[c].get('class_') == 'TerminalNodeImpl' and (
                                      G.nodes[c].get('value') == '\\}' or G.nodes[c].get('value') == '}')]
        if direct_closers:
            closers = direct_closers
        else:
            closers = [d for d in HeaderCompletionModel._descendants(G, header_id)
                       if G.nodes[d].get('class_') == 'TerminalNodeImpl' and (
                                   G.nodes[d].get('value') == '\\}' or G.nodes[d].get('value') == '}')]

        for c in closers:
            if c in G:
                G.remove_node(c)

        HeaderCompletionModel._add_terminal(G, header_id, '}')

    def save_model(self, path: str) -> None:
        """
        Persist model weights and (optionally) encoders to disk.

        The method tries to serialize scikit-learn ``LabelEncoder`` classes if
        present; if not, it still stores the model state dict for later loading.

        Args:
            path: Directory to create or reuse for the checkpoint.

        Returns:
            None. Prints a confirmation with the target path.
        """
        os.makedirs(path, exist_ok=True)
        enc_ser = {}
        try:
            from sklearn.preprocessing import LabelEncoder
            if getattr(self, "class_encoder", None) is not None and isinstance(self.class_encoder, LabelEncoder):
                enc_ser["class_encoder_classes"] = self.class_encoder.classes_.tolist()
            if getattr(self, "value_encoder", None) is not None and isinstance(self.value_encoder, LabelEncoder):
                enc_ser["value_encoder_classes"] = self.value_encoder.classes_.tolist()
        except Exception:
            pass

        torch.save({
            "model_state": self.state_dict(),
            "class_encoder": getattr(self, "class_encoder", None),
            "value_encoder": getattr(self, "value_encoder", None),
            "encoders_serialized": enc_ser,
        }, os.path.join(path, "header_completion_model.pt"))
        print(f"HeaderCompletionModel saved → {path}")

    def load_model(self, path: str) -> None:
        """
        Load model weights and, if available, serialized encoders from disk.

        The loader is tolerant to different PyTorch versions and attempts to
        register safe globals for scikit-learn encoders, reconstructing them
        when serialized class lists are present.

        Args:
            path: Directory containing ``header_completion_model.pt``.

        Returns:
            None. Moves the model to ``self.device`` and prints a confirmation.
        """
        ckpt_path = os.path.join(path, "header_completion_model.pt")
        try:
            try:
                from torch.serialization import add_safe_globals
                try:
                    from sklearn.preprocessing import LabelEncoder
                    add_safe_globals([LabelEncoder])
                except Exception:
                    pass
            except Exception:
                pass
            ckpt = torch.load(ckpt_path, map_location=self.device)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if "model_state" in ckpt:
            self.load_state_dict(ckpt["model_state"])
        else:
            self.load_state_dict(ckpt)

        enc_ser = ckpt.get("encoders_serialized") or {}
        if enc_ser:
            try:
                from sklearn.preprocessing import LabelEncoder
                import numpy as np
                if enc_ser.get("class_encoder_classes") is not None:
                    le_c = LabelEncoder()
                    le_c.classes_ = np.array(enc_ser["class_encoder_classes"], dtype=object)
                    self.class_encoder = le_c
                if enc_ser.get("value_encoder_classes") is not None:
                    le_v = LabelEncoder()
                    le_v.classes_ = np.array(enc_ser["value_encoder_classes"], dtype=object)
                    self.value_encoder = le_v
            except Exception:
                pass

        if getattr(self, "class_encoder", None) is None and ckpt.get("class_encoder") is not None:
            self.class_encoder = ckpt.get("class_encoder")
        if getattr(self, "value_encoder", None) is None and ckpt.get("value_encoder") is not None:
            self.value_encoder = ckpt.get("value_encoder")

        self.to(self.device)
        print(f"HeaderCompletionModel loaded ← {path}")

    def _label_nodes(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        (Placeholder) Produce supervision labels for nodes of a training graph.

        Implementations should map graph nodes to a tensor suitable for the task,
        e.g., a float tensor of shape ``[num_nodes]`` with binary labels indicating
        whether a node is a positive header target.

        Args:
            graph: AST graph used to compute labels.

        Returns:
            A ``torch.Tensor`` of labels aligned with the model's node ordering.
        """
        pass

    def _train_epoch(self, dataset: List[Data], epoch: int) -> None:
        """
        (Placeholder) Run one training epoch over a dataset of PyG ``Data`` items.

        Implementations typically:
          * encode node features,
          * run the GNN,
          * compute task-specific losses, and
          * step the optimizer.

        Args:
            dataset: List of PyG graphs prepared by ``_graph_to_pyg``.
            epoch: Current epoch index (0-based or 1-based, implementation-defined).

        Returns:
            None.
        """
        pass

    def predict_subgraph(self, graph_path: str, node_embeddings: torch.Tensor) -> List[int]:
        """
        (Placeholder) Predict a set of node ids forming a task-specific subgraph.

        This helper is intended for scenarios where the GNN embeddings are already
        available (e.g., cached) and only post-processing/selection is required.

        Args:
            graph_path: Path to the input JSON graph (for node id mapping).
            node_embeddings: Tensor of node embeddings aligned with the graph.

        Returns:
            A list of selected node ids (possibly empty).
        """
        pass
