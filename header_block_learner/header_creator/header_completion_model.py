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
    type: Optional[str] = None
    name: Optional[str] = None


class HeaderCompletionModel(AbstractGraphLearner):
    """
    Feltételes beszúrási-hely prediktor és opcionális AST-építő.

    - A GNN kódoló az `AbstractGraphLearner`-ből jön (class/value encoderek + GCN).
    - `cond_head`: olyan MLP, ami [node_emb ; cond_emb] → logit kimenetet ad.
      A `cond_emb` a felhasználó által megadott (type, name) tokenekből készül.
    - Tanítás: (teljes, redukált) párokból származtatott címkékkel.
        * Pozitív: azon `HeaderTypeDeclarationContext` node-ok a redukált gráfban,
          amelyek alatt mező hiányzik a teljeshez képest (azonos node-id alapján).
        * Negatív: a többi header node.
    """

    # ---- Kötelező/absztrakt metódusok a bázisosztályból: nem szükségesek a feladathoz → pass ----
    def _label_nodes(self, graph: nx.DiGraph) -> torch.Tensor:
        pass

    def _train_epoch(self, dataset: List[Data], epoch: int) -> None:
        pass

    def predict_subgraph(self, graph_path: str, node_embeddings: torch.Tensor) -> List[int]:
        pass

    def __init__(self, hidden_dim: int = 64, device: str = "cpu",
                 gnn_layers: int = 3, gnn_dropout: float = 0.10):
        super().__init__(hidden_dim=hidden_dim, device=device,
                         gnn_layers=gnn_layers, gnn_dropout=gnn_dropout)
        # Feltételes fej: [node_emb ; cond_emb] → 1
        self.cond_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.10),
            nn.Linear(self.hidden_dim, 1)
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    # ----------------------------
    # UTIL: JSON <-> networkx
    # ----------------------------
    @staticmethod
    def load_graph_json(path: str) -> nx.DiGraph:
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
        """Kompatibilitási wrapper: a bázisban is lehet hasonló segédfüggvény."""
        return self.load_graph_json(path)

    @staticmethod
    def dump_graph_json(G: nx.DiGraph, path: str) -> str:
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

    # ----------------------------
    # HEADER / FIELD lokalizálás
    # ----------------------------
    @staticmethod
    def _header_nodes(G: nx.DiGraph) -> List[int]:
        return [n for n in G.nodes if G.nodes[n].get("class_") == "HeaderTypeDeclarationContext"]

    @staticmethod
    def _struct_fields_under(G: nx.DiGraph, header_id: int) -> List[int]:
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
        q, vis = [nid], set()
        while q:
            cur = q.pop(0)
            for nxt in G.successors(cur):
                if nxt not in vis:
                    vis.add(nxt)
                    q.append(nxt)
                    yield nxt

    @staticmethod
    def _first_value_under(G: nx.DiGraph, nid: int, class_path_prefix: Optional[Tuple[str, ...]] = None) -> Optional[
        str]:
        """Heurisztika: keressünk `TerminalNodeImpl.value`-t az al-fában.
        Ha `class_path_prefix` meg van adva, akkor előnyben részesítjük az olyan
        útvonalakat, ahol a csomópontok osztálylánca ilyen prefixszel kezdődik
        (pl. ("TypeNameContext",))."""
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
        # Típus: próbáljuk TypeName/PrefixedType ágakon a legelső terminált
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

    # ----------------------------
    # PÁR-ADATOK FELDOLGOZÁSA
    # ----------------------------
    @staticmethod
    def _match_pairs_in_dir(pairs_dir: str) -> List[Tuple[str, str]]:
        """Párok kigyűjtése: `*_reduced.json` ↔ azonos név `_reduced` nélkül."""
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
        """(header_id, missing_field_id_in_full) listát ad vissza."""
        miss: List[Tuple[int, int]] = []
        for h in HeaderCompletionModel._header_nodes(fullG):
            if h not in redG:
                # Ha a header node teljesen hiányzik a redukáltból, átugorjuk
                continue
            full_fields = set(HeaderCompletionModel._struct_fields_under(fullG, h))
            red_fields = set(HeaderCompletionModel._struct_fields_under(redG, h))
            missing = [fid for fid in full_fields if fid not in red_fields]
            for fid in missing:
                miss.append((h, fid))
        return miss

    # ----------------------------
    # ENCODING A FELTÉTELHEZ
    # ----------------------------
    def _encode_condition(self, specs: List[FieldSpec]) -> torch.Tensor:
        """Egy vagy több mezőspecifikációt vektorozunk → [hidden_dim].
        Heurisztika: type/name token embeddingek átlaga, majd átlag a mezők között is.
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
                enc_t = torch.as_tensor(enc, dtype=torch.float32, device=self.device).unsqueeze(1)  # [T,1]
                ve = self.value_autoencoder(enc_t)  # [T, hidden]
                val_emb = ve.mean(dim=0)  # [hidden]
            vals.append(val_emb)
        return torch.stack(vals, dim=0).mean(dim=0)

    # ----------------------------
    # TANÍTÁS PÁROKON
    # ----------------------------
    def fit_on_pairs(self, pairs: List[Tuple[str, str]], epochs: int = 20,
                     prune_step: float = 0.05, prune_max_ratio: float = 0.30) -> None:
        """Tanítás (teljes_path, redukált_path) párokon.

        Megjegyzés: az encoder(eke)t a *teljes és redukált* fájlok unióján tanítjuk.
        """
        # 1) Encoder tanítás: minden érintett fájl
        all_files: List[str] = []
        for full_p, red_p in pairs:
            all_files.extend([full_p, red_p])
        self.fit_encoders(all_files)

        # 2) Epoch-ok
        for epoch in range(epochs):
            ratio = min(prune_step * (epoch + 1), prune_max_ratio)
            total_loss = 0.0
            total_pos = 0
            total_mask = 0

            for full_p, red_p in pairs:
                fullG = self._load_graph_from_json(full_p)
                redG = self._load_graph_from_json(red_p)

                # opcionális augmentáció a redukálton (stabilitás): levelek/subtree törlés
                if epoch > 0:
                    if epoch < 3:
                        redG = self._delete_random_leaves(redG, ratio)
                    else:
                        redG = self._delete_random_subtrees(redG, ratio)

                # Hiányzó mezők és címkék
                miss = self._missing_fields(fullG, redG)  # [(header_id, field_id_in_full), ...]
                if not miss:
                    continue

                # Kinyerjük a hiányzó mezők tokenjeit → condition
                cond_specs: List[FieldSpec] = []
                for (_, fid) in miss:
                    cond_specs.append(self._extract_field_tokens(fullG, fid))
                cond_vec = self._encode_condition(cond_specs)  # [hidden]

                # PyG adatok és node-embedingek a redukált gráfról
                data = self._graph_to_pyg(redG)
                x = self._encode_node_features(data._raw_node_attrs)
                emb = self.gnn(x, data.edge_index)  # [N, hidden]

                # Címkék: csak a HeaderTypeDeclarationContext csomópontok számitanak
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

                # Ha nincs header a maszkban, lépjünk tovább
                if is_header.sum().item() == 0:
                    continue

                # Feltétel vektor broadcasting a csomópontokra
                cond = cond_vec.unsqueeze(0).expand(emb.size(0), -1)  # [N, hidden]
                logits = self.cond_head(torch.cat([emb, cond], dim=1)).squeeze(-1)  # [N]

                # Maszk alkalmazása
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
        pairs = self._match_pairs_in_dir(pairs_dir)
        if not pairs:
            print(f"No pairs found in: {pairs_dir}")
            return
        self.fit_on_pairs(pairs, epochs=epochs, prune_step=prune_step, prune_max_ratio=prune_max_ratio)

    # ----------------------------
    # PREDIKCIÓ: beszúrási hely pontozása
    # ----------------------------
    @torch.no_grad()
    def score_headers(self, graph_path: str, specs: List[FieldSpec]) -> List[Tuple[int, float]]:
        """Visszaadja az összes `HeaderTypeDeclarationContext` node-ra a pontszámot.
        Kimenet: [(header_node_id, score), ...] score ∈ [0,1].
        """
        G = self._load_graph_from_json(graph_path)
        data = self._graph_to_pyg(G)
        x = self._encode_node_features(data._raw_node_attrs)
        emb = self.gnn(x, data.edge_index)  # [N, hidden]
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

    # ----------------------------
    # AST-ÉPÍTÉS: mező beszúrása
    # ----------------------------
    @staticmethod
    def _ensure_field_list(G: nx.DiGraph, header_id: int) -> int:
        """Ha nincs `StructFieldListContext` gyerek, létrehozzuk és visszaadjuk az id-ját."""
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
        base = max(int(n) for n in G.nodes if isinstance(n, int)) + 1 if len(G) else 1
        return list(range(base, base + k))

    @staticmethod
    def _add_terminal(G: nx.DiGraph, parent: int, value: str) -> int:
        nid = HeaderCompletionModel._new_id(G, 1)[0]
        G.add_node(nid, id=nid, nodeId=nid, label="syn", line=-1, start=-1, end=-1,
                   value=value, class_="TerminalNodeImpl")
        G.add_edge(parent, nid)
        return nid

    @staticmethod
    def _add_node(G: nx.DiGraph, parent: int, cls: str, value: Optional[Any] = None) -> int:
        nid = HeaderCompletionModel._new_id(G, 1)[0]
        G.add_node(nid, id=nid, nodeId=nid, label="syn", line=-1, start=-1, end=-1,
                   value=value, class_=cls)
        if parent is not None:
            G.add_edge(parent, nid)
        return nid

    @staticmethod
    def _build_field_subtree(G: nx.DiGraph, list_id: int, spec: FieldSpec) -> int:
        """Minimális P4 mező AST felépítése a mintához hasonlóan.

        Struktúra:
            StructFieldContext
              ├─ TypeRefContext → TypeNameContext → PrefixedTypeContext → Type_or_idContext → Terminal(type)
              ├─ NameContext → NonTypeNameContext → Type_or_idContext → Terminal(name)
              └─ Terminal(';')
        """
        f_id = HeaderCompletionModel._add_node(G, list_id, "StructFieldContext")

        # Type-ág
        tref = HeaderCompletionModel._add_node(G, f_id, "TypeRefContext")
        tname = HeaderCompletionModel._add_node(G, tref, "TypeNameContext")
        tpre = HeaderCompletionModel._add_node(G, tname, "PrefixedTypeContext")
        ttoi = HeaderCompletionModel._add_node(G, tpre, "Type_or_idContext")
        HeaderCompletionModel._add_terminal(G, ttoi, spec.type or "<UNK_TYPE>")

        # Name-ág
        nname = HeaderCompletionModel._add_node(G, f_id, "NameContext")
        nnon = HeaderCompletionModel._add_node(G, nname, "NonTypeNameContext")
        ntoi = HeaderCompletionModel._add_node(G, nnon, "Type_or_idContext")
        HeaderCompletionModel._add_terminal(G, ntoi, spec.name or "<UNK_NAME>")

        # ;
        HeaderCompletionModel._add_terminal(G, f_id, ";")
        return f_id

    @staticmethod
    def _find_header_by_name(G: nx.DiGraph, header_name: str) -> List[int]:
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
        """Gráf kiegészítése a megadott specifikáció szerint.

        `additions` kulcsa lehet header **név** (str) vagy **node_id** (int).
        Értéke: list of {"type": str, "name": str}.
        Ha a header név több headerre illeszkedik, a modell pontoz és a legjobb
        headerbe szúrunk.
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

        # 2) Név szerinti kiválasztás pontozással (ha több jelölt van)
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

        # 3) Beszúrás
        for hid, specs in by_id.items():
            if hid not in G:
                continue
            list_id = self._ensure_field_list(G, hid)
            for sp in specs:
                self._build_field_subtree(G, list_id, sp)

        out = output_path or (os.path.splitext(graph_path)[0] + ".completed.json")
        return self.dump_graph_json(G, out)

    # ----------------------------
    # Mentés/Betöltés
    # ----------------------------
    def save_model(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        enc_ser = {}
        try:
            from sklearn.preprocessing import LabelEncoder  # type: ignore
            if getattr(self, "class_encoder", None) is not None and isinstance(self.class_encoder, LabelEncoder):
                enc_ser["class_encoder_classes"] = self.class_encoder.classes_.tolist()
            if getattr(self, "value_encoder", None) is not None and isinstance(self.value_encoder, LabelEncoder):
                enc_ser["value_encoder_classes"] = self.value_encoder.classes_.tolist()
        except Exception:
            pass

        torch.save({
            "model_state": self.state_dict(),
            # Visszafelé kompatibilitás: a nyers objektumokat is eltesszük,
            # de ezek 2.6 alatt csak allowlist-tel fognak betölteni.
            "class_encoder": getattr(self, "class_encoder", None),
            "value_encoder": getattr(self, "value_encoder", None),
            "encoders_serialized": enc_ser,
        }, os.path.join(path, "header_completion_model.pt"))
        print(f"HeaderCompletionModel saved → {path}")

    def load_model(self, path: str) -> None:
        ckpt_path = os.path.join(path, "header_completion_model.pt")

        # 1) Próbáljuk safe allowlist-tel (PyTorch 2.6 default: weights_only=True)
        try:
            try:
                from torch.serialization import add_safe_globals  # PyTorch 2.6+
                try:
                    from sklearn.preprocessing import LabelEncoder  # type: ignore
                    add_safe_globals([LabelEncoder])
                except Exception:
                    pass
            except Exception:
                pass
            ckpt = torch.load(ckpt_path, map_location=self.device)
        except Exception:
            # 2) Fallback: weights_only=False (CSAK megbízható checkpointnál!)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Súlyok
        if "model_state" in ckpt:
            self.load_state_dict(ckpt["model_state"])
        else:
            self.load_state_dict(ckpt)

        # Encoderek visszaépítése (serialized -> LabelEncoder)
        enc_ser = ckpt.get("encoders_serialized") or {}
        if enc_ser:
            try:
                from sklearn.preprocessing import LabelEncoder  # type: ignore
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

        # Ha nincs serialized és az allowlist miatt sikerült a nyers objektumot betölteni, azt használjuk
        if getattr(self, "class_encoder", None) is None and ckpt.get("class_encoder") is not None:
            self.class_encoder = ckpt.get("class_encoder")
        if getattr(self, "value_encoder", None) is None and ckpt.get("value_encoder") is not None:
            self.value_encoder = ckpt.get("value_encoder")

        self.to(self.device)
        print(f"HeaderCompletionModel loaded ← {path}")
