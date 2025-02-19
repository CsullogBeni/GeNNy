import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Create a directed graph
G = nx.DiGraph()

'''# Add nodes with custom fields
G.add_node("A", nodeId=1, line=10, start=5, end=15, class_="Node", value=42)
G.add_node("B", nodeId=2, line=20, start=10, end=20, class_="Node", value=None)
G.add_node("C", nodeId=3, line=30, start=15, end=25, class_="Node", value=24)

# Add directed edges
G.add_edge("A", "B")
G.add_edge("B", "C")'''

G.add_node("0", nodeId=0, line=1, start=1, end=1, class_="AssignmentOrMethodCallStatementContext", value=None)
G.add_node("1", nodeId=1, line=1, start=1, end=1, class_="LvalueContext", value=None)
G.add_node("2", nodeId=2, line=1, start=1, end=1, class_="LvalueContext", value=None)
G.add_node("3", nodeId=3, line=1, start=1, end=1, class_="LvalueContext", value=None)
G.add_node("4", nodeId=4, line=1, start=1, end=1, class_="PrefixedNonTypeNameContext", value=None)
G.add_node("5", nodeId=5, line=1, start=1, end=1, class_="NonTypeNameContext", value=None)
G.add_node("6", nodeId=6, line=1, start=1, end=1, class_="Type_or_idContext", value=None)
G.add_node("7", nodeId=7, line=1, start=1, end=1, class_="TerminalNodeImpl", value="hdr")
G.add_node("8", nodeId=8, line=1, start=1, end=1, class_="TerminalNodeImpl", value=".")
G.add_node("9", nodeId=9, line=1, start=1, end=1, class_="NameContext", value=None)
G.add_node("10", nodeId=10, line=1, start=1, end=1, class_="NonTypeNameContext", value=None)
G.add_node("11", nodeId=11, line=1, start=1, end=1, class_="Type_or_idContext", value=None)
G.add_node("12", nodeId=12, line=1, start=1, end=1, class_="TerminalNodeImpl", value="ethernet")
G.add_node("13", nodeId=13, line=1, start=1, end=1, class_="TerminalNodeImpl", value=".")
G.add_node("14", nodeId=14, line=1, start=1, end=1, class_="NameContext", value=None)
G.add_node("15", nodeId=15, line=1, start=1, end=1, class_="NonTypeNameContext", value=None)
G.add_node("16", nodeId=16, line=1, start=1, end=1, class_="Type_or_idContext", value=None)
G.add_node("17", nodeId=17, line=1, start=1, end=1, class_="TerminalNodeImpl", value="dstAddr")
G.add_node("18", nodeId=18, line=1, start=1, end=1, class_="TerminalNodeImpl", value="=")
G.add_node("19", nodeId=19, line=1, start=1, end=1, class_="ExpressionContext", value=None)
G.add_node("20", nodeId=20, line=1, start=1, end=1, class_="NonTypeNameContext", value=None)
G.add_node("21", nodeId=21, line=1, start=1, end=1, class_="Type_or_idContext", value=None)
G.add_node("22", nodeId=22, line=1, start=1, end=1, class_="TerminalNodeImpl", value="dstAddr")
G.add_node("23", nodeId=23, line=1, start=1, end=1, class_="TerminalNodeImpl", value=";")

G.add_edge("0", "1")
G.add_edge("0", "18")
G.add_edge("0", "19")
G.add_edge("0", "23")

G.add_edge("1", "2")
G.add_edge("1", "13")
G.add_edge("1", "14")

G.add_edge("2", "3")
G.add_edge("2", "8")
G.add_edge("2", "9")

G.add_edge("3", "4")
G.add_edge("4", "5")
G.add_edge("5", "6")
G.add_edge("6", "7")

G.add_edge("9", "10")
G.add_edge("10", "11")
G.add_edge("11", "12")

G.add_edge("14", "15")
G.add_edge("15", "16")
G.add_edge("16", "17")

G.add_edge("19", "20")
G.add_edge("19", "22")
G.add_edge("20", "21")
G.add_edge("21", "22")

with open("graph.pkl", "wb") as f:
    pickle.dump(G, f)
with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

# Draw the graph
pos = nx.spring_layout(G)

# Draw node labels with custom fields
node_labels = {node: f"NodeId: {G.nodes[node]['nodeId']}\nLine: {G.nodes[node]['line']}\nStart: {G.nodes[node]['start']}\nEnd: {G.nodes[node]['end']}\nClass: {G.nodes[node]['class_']}\nValue: {G.nodes[node]['value'] if G.nodes[node]['value'] is not None else 'None'}" for node in G.nodes}

# Draw nodes as boxes
for node in G.nodes:
    x, y = pos[node]
    plt.text(x, y, node_labels[node], ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round'))

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='k', arrowsize=100, arrowstyle='->')
'''for edge in G.edges:
    x1, y1 = pos[edge[0]]
    x2, y2 = pos[edge[1]]
    plt.plot([x1, x2], [y1, y2], 'k-')
'''
plt.axis('off')
plt.show()

classes = list(set(nx.get_node_attributes(G, 'class_').values()))
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(np.array(classes).reshape(-1, 1))

'''def node_to_vector(node, attrs):
    """Átalakít egy csomópontot numerikus vektorrá."""
    node_id = attrs.get("nodeId", 0)
    line = attrs.get("line", 0)
    start = attrs.get("start", 0)
    end = attrs.get("end", 0)
    value = attrs.get("value", "None")  # Ha None, akkor stringgé alakítjuk
    class_ = attrs.get("class_", "Unknown")

    # One-hot encoding a class_ mezőhöz
    class_encoded = encoder.transform([[class_]])[0]

    # Ha value egy szám, megtartjuk, ha nem, akkor egy konstans (-1) értéket adunk
    value_numeric = float(value) if value and value.replace('.', '', 1).isdigit() else -1

    return np.concatenate((np.array([node_id, line, start, end, value_numeric]), np.array([class_encoded])))'''

def node_to_vector(node, attrs):
    node_id = attrs.get("nodeId", 0)
    line = attrs.get("line", 0)
    start = attrs.get("start", 0)
    end = attrs.get("end", 0)
    value = attrs.get("value", "None")  # Ha None, akkor stringgé alakítjuk
    class_ = attrs.get("class_", "Unknown")

    # One-hot encoding a class_ mezőhöz
    class_encoded = encoder.transform([[class_]])[0].toarray()[0]

    # Ha value egy szám, megtartjuk, ha nem, akkor egy konstans (-1) értéket adunk
    value_numeric = float(value) if value and value.replace('.', '', 1).isdigit() else -1

    return np.concatenate((np.array([node_id, line, start, end, value_numeric]), np.array(class_encoded)))


# Vektorizáljuk az összes csomópontot
node_vectors = {node: node_to_vector(node, G.nodes[node]) for node in G.nodes}

# A vektorizált csomópontok kiírása
'''for node, vector in node_vectors.items():
    print(f"Node {node}: {vector}")

print()
print()'''

# 2. Szomszédsági mátrix előállítása
adj_matrix = nx.to_numpy_array(G)
#edge_index = torch.tensor(np.array(G.edges).T, dtype=torch.long)
# Get the edge indices as a NumPy array
edge_array = nx.to_numpy_array(G)

# Get the indices of the edges
edge_index = np.where(edge_array)

# Convert the edge indices to a PyTorch tensor
edge_index = torch.tensor(edge_index, dtype=torch.long)

# 3. Csomópont jellemzők
'''node_features = np.array([node_to_vector(n, G.nodes[n]) for n in G.nodes])
node_features = torch.tensor(node_features, dtype=torch.float)'''
node_features = np.array([node_to_vector(n, G.nodes[n]) for n in G.nodes], dtype=np.float64)
node_features = torch.tensor(node_features, dtype=torch.float)

# 4. PyTorch Geometric adatformátum létrehozása
data = Data(x=node_features, edge_index=edge_index)


# 5. GNN modell definiálása
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# 6. Modell inicializálása
model = GNN(in_channels=node_features.shape[1], hidden_channels=16, out_channels=node_features.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


# 7. Tanítási ciklus (alap rekonstrukció)
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, data.x)  # Rekonstrukciós loss
    loss.backward()
    optimizer.step()
    return loss.item()


# 8. Gráf módosítása (csomópontok és élek törlése)
def modify_graph(data, removal_ratio=0.2):
    num_edges = data.edge_index.shape[1]
    num_remove = int(removal_ratio * num_edges)
    mask = torch.ones(num_edges, dtype=torch.bool)
    remove_indices = np.random.choice(num_edges, num_remove, replace=False)
    mask[remove_indices] = False
    new_edge_index = data.edge_index[:, mask]
    return Data(x=data.x, edge_index=new_edge_index)


# 9. Tanítás törölt élekkel
original_data = data
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Gráf módosítása 20. epoch után
    if epoch == 20:
        data = modify_graph(data, removal_ratio=0.2)


# 10. Eredeti és rekonstruált gráf megjelenítése
def visualize_graph(original_data, modified_data, reconstructed_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Helper function for drawing graphs
    def draw_graph(ax, data, title):
        G = nx.Graph()
        G.add_edges_from(data.edge_index.T.tolist())
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)
        ax.set_title(title)

    draw_graph(axes[0], original_data, "Original Graph")
    draw_graph(axes[1], modified_data, "Incomplete Graph")
    draw_graph(axes[2], reconstructed_data, "Reconstructed Graph")

    plt.show()

# 11. Rekonstruált gráf előállítása
def reconstruct_graph(data):
    model.eval()
    with torch.no_grad():
        reconstructed_x = model(data)
    return Data(x=reconstructed_x, edge_index=data.edge_index)

# 12. Megjelenítés
'''reconstructed_data = reconstruct_graph(data)
visualize_graph(original_data, data, reconstructed_data)'''

reconstructed_data = reconstruct_graph(data)
def draw_graph(ax, data, title, with_arrows=True):
    G = nx.DiGraph()
    G.add_edges_from(data.edge_index.T.tolist())
    pos = nx.spring_layout(G)
    if with_arrows:
        nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', connectionstyle='arc3,rad=0.1')
    else:
        nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    ax.set_title(title)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
draw_graph(axes[0], original_data, "Original Graph", with_arrows=False)
draw_graph(axes[1], data, "Incomplete Graph", with_arrows=False)
draw_graph(axes[2], reconstructed_data, "Reconstructed Graph", with_arrows=True)
plt.show()
