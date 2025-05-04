import json
from pathlib import Path
from typing import Dict, List
import torch
from torch_geometric.data import Data, DataLoader
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os


def load_ast(json_path: Path) -> Dict:
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def build_child_map(edges: List[Dict[str, int]]) -> Dict[int, List[int]]:
    child_map: Dict[int, List[int]] = defaultdict(list)
    for e in edges:
        child_map[e["source"].__int__()].append(e["target"].__int__())
    return child_map


def find_empty_else_blocks(ast: Dict) -> List[int]:
    nodes = ast["nodes"]
    edges = ast["edges"]
    node_by_id = {n["nodeId"]: n for n in nodes}
    children_of = build_child_map(edges)

    empty_else_node_ids = []

    for cond in (n for n in nodes if n.get("class_") == "ConditionalStatementContext"):
        cond_id = cond["nodeId"]
        kids = children_of.get(cond_id, [])

        else_ids = [k for k in kids if node_by_id[k].get("value") == "else"]
        stmt_ids = [k for k in kids if node_by_id[k].get("class_") == "StatementContext"]

        for else_id in else_ids:
            for stmt_id in stmt_ids:
                if else_id > stmt_id:
                    continue

                block_ids = [
                    c for c in children_of.get(stmt_id, [])
                    if node_by_id[c].get("class_") == "BlockStatementContext"
                ]

                for block_id in block_ids:
                    stat_ids = [
                        c for c in children_of.get(block_id, [])
                        if node_by_id[c].get("class_") == "StatOrDeclListContext"
                    ]
                    for stat_id in stat_ids:
                        if not children_of.get(stat_id):
                            empty_else_node_ids.append(else_id)
    return empty_else_node_ids


def create_node_features(nodes: List[Dict], label_else_ids: List[int]) -> (torch.Tensor, List[int], List[int]):
    class_labels = [n.get("class_", "") for n in nodes]
    value_labels = [(n.get("value") if n.get("class_") == "TerminalNodeImpl" else "") for n in nodes]

    class_enc = LabelEncoder()
    value_enc = LabelEncoder()

    class_ids = class_enc.fit_transform(class_labels)
    value_ids = value_enc.fit_transform(value_labels)

    x = []
    y = []
    mask = []
    target_node_ids = []

    for i, n in enumerate(nodes):
        vec = [class_ids[i], value_ids[i]]
        x.append(vec)

        if n.get("value") == "else":
            node_id = n["nodeId"]
            label = 1 if node_id in label_else_ids else 0
            y.append(label)
            mask.append(True)
            target_node_ids.append(node_id)
        else:
            y.append(-1)
            mask.append(False)

    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), torch.tensor(mask,
                                                                                              dtype=torch.bool), target_node_ids


def build_data(ast_path: Path) -> Data:
    ast = load_ast(ast_path)
    empty_else_ids = find_empty_else_blocks(ast)
    nodes = ast["nodes"]
    edges = ast["edges"]

    x, y, mask, target_node_ids = create_node_features(nodes, empty_else_ids)

    edge_index = torch.tensor([[e["source"], e["target"]] for e in edges], dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=mask)
    data.target_node_ids = target_node_ids
    data.file_name = ast_path.name
    return data


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)  # 2 osztály: üres vagy nem üres

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred[data.train_mask] == data.y[data.train_mask]).sum())
            total += int(data.train_mask.sum())
    return correct / total if total > 0 else 0.0


def predict(model, dataset):
    model.eval()
    print("\n--- Prediction Results ---")
    with torch.no_grad():
        for data in dataset:
            out = model(data)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            for i, is_target in enumerate(data.train_mask):
                if is_target:
                    label = pred[i].item()
                    confidence = probs[i][label].item()
                    node_id = data.x[i].tolist()  # nem a nodeId, hanem jellemző; ha kell, tároljuk külön!
                    print(
                        f"{data.file_name}: else node idx {i} => predicted: {label} ({'empty' if label == 1 else 'non-empty'}, confidence={confidence:.2f})")


if __name__ == "__main__":
    data_dir = Path("data")
    graph_files = list(data_dir.glob("*.json"))
    dataset = [build_data(path) for path in graph_files]
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    input_dim = 2  # class_ és value embeddingek integer ID-ként
    model = GCN(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(200):
        loss = train(model, loader, optimizer, criterion)
        acc = evaluate(model, loader)
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Előrejelzés
    predict(model, dataset)
