import json
import dictdiffer
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


original_data = load_json("assignment.json")
renamed_data = load_json("assignment_renamed.json")


def build_graph(data):
    G = nx.DiGraph()
    for node in data["nodes"]:
        attributes = {k: v for k, v in node.items() if k != "nodeId"}
        if node["class_"] == "TerminalNodeImpl":
            attributes["value"] = node["value"]
        else:
            attributes.pop("value", None)
        G.add_node(node["nodeId"], **attributes)
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"])
    return G


graph_original = build_graph(original_data)
graph_renamed = build_graph(renamed_data)


class NodeEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim=16):
        super(NodeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, node_ids):
        return self.embedding(node_ids)


num_nodes = max(graph_original.nodes) + 1
node_embedding = NodeEmbedding(num_nodes)


class NameChangeModel(nn.Module):
    def __init__(self, embedding_dim=16):
        super(NameChangeModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))


model = NameChangeModel()
optimizer = optim.Adam(list(model.parameters()) + list(node_embedding.parameters()), lr=0.01)
loss_fn = nn.BCELoss()


def extract_features(graph, embedding_layer):
    features = []
    targets = []
    node_ids = []
    for node, data in graph.nodes(data=True):
        if data.get("class_") == "TerminalNodeImpl":
            node_ids.append(node)
            target_value = 1.0 if data.get("value") == "destinationAddress" else 0.0
            targets.append(torch.tensor([target_value]))
    return torch.tensor(node_ids, dtype=torch.long), torch.stack(targets)


input_nodes, target_labels = extract_features(graph_original, node_embedding)
_, renamed_labels = extract_features(graph_renamed, node_embedding)


def train_model(model, embedding_layer, optimizer, loss_fn, input_nodes, target_data, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = embedding_layer(input_nodes)
        output = model(embeddings)
        loss = loss_fn(output, target_data)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")


train_model(model, node_embedding, optimizer, loss_fn, input_nodes, renamed_labels)


def rename_graph(model, embedding_layer, graph):
    new_graph = graph.copy()
    for node, data in new_graph.nodes(data=True):
        if data.get("class_") == "TerminalNodeImpl":
            input_vector = embedding_layer(torch.tensor([node], dtype=torch.long)).squeeze(0)
            prediction = model(input_vector).item()
            if prediction > 0.75:
                data["value"] = "destinationAddress"
    return new_graph


renamed_graph = rename_graph(model, node_embedding, graph_original)

'''print(original_data)'''
renamed_data = {'nodes': [], 'edges': []}
for node in renamed_graph.nodes:
    current = renamed_graph.nodes[node]
    current['nodeId'] = node
    renamed_data['nodes'].append(current)
for edge in renamed_graph.edges:
    renamed_data['edges'].append({'source': edge[0], 'target': edge[1]})
'''print(renamed_data)'''

print("Differences:")
for index in range(len(original_data['nodes'])):
    try:
        difference = 'Node ' + str(index) + ': '
        current_original = original_data['nodes'][index]
        current_renamed = renamed_data['nodes'][index]
        if current_original['nodeId'] != current_renamed['nodeId']:
            difference += 'nodeId: ' + str(current_original['nodeId']) + ' -> ' + str(current_renamed['nodeId']) + ', '
        if current_original['line'] != current_renamed['line']:
            difference += 'line: ' + str(current_original['line']) + ' -> ' + str(current_renamed['line']) + ', '
        if current_original['start'] != current_renamed['start']:
            difference += 'start: ' + str(current_original['start']) + ' -> ' + str(current_renamed['start']) + ', '
        if current_original['end'] != current_renamed['end']:
            difference += 'end: ' + str(current_original['end']) + ' -> ' + str(current_renamed['end']) + ', '
        if current_original['class_'] != current_renamed['class_']:
            difference += 'class_: ' + str(current_original['class_']) + ' -> ' + str(current_renamed['class_']) + ', '
        try:
            if current_original['value'] != current_renamed['value']:
                difference += 'value: ' + str(current_original['value']) + ' -> ' + str(current_renamed['value']) + ', '
        except KeyError:
            pass
        if difference != 'Node ' + str(index) + ': ':
            print(difference)
    except Exception as e:
        print(e)
        continue
