import os
import json
import glob
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim


class ParamOrderModel:
    def __init__(self, data_dir, embedding_dim=64, lr=0.001, epochs=15):
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        self.epochs = epochs

        self.node_embedding = nn.Embedding(10_000, embedding_dim)  # nagy elemszám, padding miatt
        self.model = self.OrderPredictionModel(embedding_dim)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.node_embedding.parameters()), lr=lr
        )
        self.loss_fn = nn.MSELoss()

    class OrderPredictionModel(nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.fc(x).squeeze(-1)

    @staticmethod
    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def build_graph(data):
        G = nx.DiGraph()
        for node in data['nodes']:
            node_id = node['nodeId']
            attributes = node.copy()
            G.add_node(node_id, **attributes)
        for edge in data.get('edges', []):
            G.add_edge(edge['source'], edge['target'])
        return G

    @staticmethod
    def get_ordered_parameters(graph):
        params = []
        for node_id, data in graph.nodes(data=True):
            if data.get("class_") == "ParameterContext":
                params.append((data["start"], node_id))
        return [node_id for _, node_id in sorted(params)]

    @staticmethod
    def get_param_signature(graph, node_id):
        children = list(graph.successors(node_id))
        typename = name = None
        for child in children:
            data = graph.nodes[child]
            if data["class_"] == "TypeRefContext":
                for sub in graph.successors(child):
                    sub_data = graph.nodes[sub]
                    if sub_data["class_"] == "TerminalNodeImpl":
                        typename = sub_data.get("value")
            if data["class_"] == "NameContext":
                for sub in graph.successors(child):
                    sub_data = graph.nodes[sub]
                    if sub_data["class_"] == "TerminalNodeImpl":
                        name = sub_data.get("value")
        return typename, name

    @staticmethod
    def get_target_permutation(graph_original, graph_reordered, param_nodes_orig, param_nodes_reord):
        reord_sigs = [ParamOrderModel.get_param_signature(graph_reordered, nid) for nid in param_nodes_reord]
        orig_sigs = [ParamOrderModel.get_param_signature(graph_original, nid) for nid in param_nodes_orig]
        positions = []
        for sig in orig_sigs:
            try:
                idx = reord_sigs.index(sig)
                positions.append(float(idx))
            except ValueError:
                raise ValueError(f"No match found for parameter signature: {sig}")
        return torch.tensor(positions, dtype=torch.float32)

    def get_example_pairs(self):
        files = glob.glob(os.path.join(self.data_dir, "*_generated.json"))
        pairs = []
        for f in files:
            if "reordered" in f:
                continue
            base = f.replace("_generated.json", "")
            reord = base + "_reordered_generated.json"
            if os.path.exists(reord):
                pairs.append((f, reord))
        return pairs

    def train(self):
        pairs = [("data/control_my_deparser.json", "data/control_my_deparser_reordered.json"),
                 ("data/control_my_compute_checksum.json", "data/control_my_compute_checksum_reordered.json"),
                 ("data/control_my_egress.json", "data/control_my_egress_reordered.json"),
                 ("data/control_my_ingress.json", "data/control_my_ingress_reordered.json"),
                 ("data/control_my_verify_checksum.json", "data/control_my_verify_checksum_reordered.json"),
                 ]
        for epoch in range(self.epochs):
            total_loss = 0.0
            for orig_path, reord_path in pairs:
                graph_original = self.build_graph(self.load_json(orig_path))
                graph_reordered = self.build_graph(self.load_json(reord_path))

                node_id_to_index = {nid: idx for idx, nid in enumerate(graph_original.nodes)}
                param_nodes_orig = self.get_ordered_parameters(graph_original)
                param_nodes_reord = self.get_ordered_parameters(graph_reordered)

                if not param_nodes_orig or not param_nodes_reord:
                    continue  # skip bad examples

                indices = torch.tensor([node_id_to_index[nid] for nid in param_nodes_orig], dtype=torch.long)
                targets = self.get_target_permutation(graph_original, graph_reordered, param_nodes_orig,
                                                      param_nodes_reord)

                self.optimizer.zero_grad()
                embeddings = self.node_embedding(indices)
                outputs = self.model(embeddings)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Total Loss: {total_loss:.4f}")

    def infer_permutation(self, graph_original, param_nodes_orig):
        node_id_to_index = {nid: idx for idx, nid in enumerate(graph_original.nodes)}
        indices = torch.tensor([node_id_to_index[nid] for nid in param_nodes_orig], dtype=torch.long)
        embeddings = self.node_embedding(indices)
        scores = self.model(embeddings)
        sorted_indices = torch.argsort(scores).tolist()

        print("Parameter prediction scores:")
        for nid, score in zip(param_nodes_orig, scores.tolist()):
            typename, name = self.get_param_signature(graph_original, nid)
            print(f"  Param: ({typename} {name}) -> score: {score:.4f}")

        sorted_indices = torch.argsort(scores).tolist()
        print("  Predicted sorted indices (new order):", sorted_indices)

        return sorted_indices

    def process_single_example(self, json_path):
        graph_original = self.build_graph(self.load_json(json_path))
        param_nodes_orig = self.get_ordered_parameters(graph_original)
        predicted_order = self.infer_permutation(graph_original, param_nodes_orig)

        reordered_graph = graph_original.copy()
        reordered_param_nodes = [param_nodes_orig[i] for i in predicted_order]

        param_root = None
        for node_id, data in reordered_graph.nodes(data=True):
            if data.get("class_") in {"ParameterListContext", "NonEmptyParameterListContext"}:
                param_root = node_id
                break
        if param_root is None:
            raise ValueError("ParameterListContext not found.")

        for child in list(reordered_graph.successors(param_root)):
            reordered_graph.remove_edge(param_root, child)

        def get_subtree_nodes(root):
            visited = set()
            stack = [root]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    stack.extend(reordered_graph.successors(node))
            return visited

        base_start = min(reordered_graph.nodes[nid]["start"] for nid in reordered_param_nodes) - 100
        new_starts = list(range(base_start, base_start + len(reordered_param_nodes) * 20, 20))
        new_commas = list(range(base_start + 10, base_start + len(reordered_param_nodes) * 20 - 10, 20))

        for i, node_id in enumerate(reordered_param_nodes):
            subtree_nodes = get_subtree_nodes(node_id)
            offset = new_starts[i] - reordered_graph.nodes[node_id]['start']
            for nid in subtree_nodes:
                reordered_graph.nodes[nid]['start'] += offset
            reordered_graph.add_edge(param_root, node_id)

            if i < len(reordered_param_nodes) - 1:
                comma_id = max(reordered_graph.nodes) + 1
                reordered_graph.add_node(comma_id, class_="TerminalNodeImpl", value=",", start=new_commas[i],
                                         label="syn", line=0, end=new_commas[i], nodeId=comma_id)
                reordered_graph.add_edge(param_root, comma_id)

        return reordered_graph


if __name__ == "__main__":
    from pretty_printer.pretty_printer import PrettyPrinter

    model = ParamOrderModel(data_dir="data", epochs=100)
    model.train()

    # Test
    test_graph = model.process_single_example("data/control_my_deparser.json")

    print("Predicted reordered script:")
    pp = PrettyPrinter(test_graph)
    print(pp.get_script)

    test_graph = model.process_single_example("data/control_my_compute_checksum.json")

    print("Predicted reordered script:")
    pp = PrettyPrinter(test_graph)
    print(pp.get_script)

    test_graph = model.process_single_example("data/control_my_egress.json")

    print("Predicted reordered script:")
    pp = PrettyPrinter(test_graph)
    print(pp.get_script)

    test_graph = model.process_single_example("data/control_my_ingress.json")

    print("Predicted reordered script:")
    pp = PrettyPrinter(test_graph)
    print(pp.get_script)

    test_graph = model.process_single_example("data/control_my_verify_checksum.json")

    print("Predicted reordered script:")
    pp = PrettyPrinter(test_graph)
    print(pp.get_script)
