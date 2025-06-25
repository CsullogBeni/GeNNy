"""
empty_else_finder.py  –  Üres else ágak keresése P4 AST‑ben
Használat:
    python empty_else_finder.py path/to/ast.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_ast(json_path: Path) -> Dict[str, Any]:
    """Betölti az AST‑t, visszaadja a node‑okat és az él‑listát."""
    with json_path.open(encoding="utf‑8") as f:
        data = json.load(f)
    return data


def build_child_map(edges: List[Dict[str, int]]) -> Dict[int, List[int]]:
    """`source → [target ...]` leképezés az él‑listából."""
    child_map: Dict[int, List[int]] = {}
    for e in edges:
        child_map.setdefault(e["source"], []).append(e["target"])
    return child_map


def find_empty_else_blocks(ast: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Visszaadja az üres else ágakat leíró dict‑ek listáját."""
    nodes = ast["nodes"]
    edges = ast["edges"]

    node_by_id = {n["nodeId"]: n for n in nodes}
    children_of = build_child_map(edges)

    results = []
    for cond in (n for n in nodes if n.get("class_") == "ConditionalStatementContext"):
        cond_id = cond["nodeId"]
        kids = children_of.get(cond_id, [])

        # else kulcsszó és StatementContext gyerekek
        else_ids = [k for k in kids if node_by_id[k].get("value") == "else"]
        stmt_ids = [k for k in kids if node_by_id[k].get("class_") == "StatementContext"]

        for else_id in else_ids:
            for stmt_id in stmt_ids:
                # az AST‑ben az 'else' token *megelőzi* a blokkot ⇒ else_id < stmt_id
                if else_id > stmt_id:
                    continue

                # BlockStatementContext a StatementContext alatt
                block_ids = [
                    c
                    for c in children_of.get(stmt_id, [])
                    if node_by_id[c].get("class_") == "BlockStatementContext"
                ]

                for block_id in block_ids:
                    # StatOrDeclListContext gyerek a blokkban
                    stat_ids = [
                        c
                        for c in children_of.get(block_id, [])
                        if node_by_id[c].get("class_") == "StatOrDeclListContext"
                    ]
                    for stat_id in stat_ids:
                        # nincsenek további gyerekei ⇒ üres blokknak számít
                        if not children_of.get(stat_id):
                            results.append(
                                {
                                    "conditional_nodeId": cond_id,
                                    "else_nodeId": else_id,
                                    "stmt_nodeId": stmt_id,
                                    "line": cond.get("line"),
                                }
                            )
    return results


def main() -> None:
    # json_path = Path(r"data/renamed_version_to_ipv4Version.json")
    json_path = Path(r"data/basic_p4_v2_normalized.json")
    ast = load_ast(json_path)
    empties = find_empty_else_blocks(ast)

    if not empties:
        print("Nem találtam üres else blokkot.")
    else:
        print(f"{len(empties)} üres else blokk:")
        for e in empties:
            print(
                f"  – if‑else kezdődik a forrás {e['line']}. sorában "
                f"(Conditional nodeId={e['conditional_nodeId']})"
            )


if __name__ == "__main__":
    main()
