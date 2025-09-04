from torch import nn
from torch_geometric.nn import GCNConv


class GNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 1, "num_layers >= 1 kell legyen"
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x, edge_index):
        # NINCS detach/clone: az encoder gradiense visszafolyik
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        return x
