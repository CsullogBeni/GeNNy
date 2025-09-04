from torch import nn
from torch_geometric.nn import GCNConv


class GNN(nn.Module):
    """
    Lightweight GCN backbone used as the graph encoder.

    The network stacks `num_layers` GCN layers. The first layer projects
    from the provided `in_dim` to `hidden_dim`; all subsequent layers keep
    `hidden_dim`. A ReLU activation (non-inplace for safe autograd) and an
    optional dropout follow each convolution.

    Notes:
        * This module is intentionally minimal. It does not include residuals,
          layer norms, nor explicit directionality handling — direction
          augmentation should be done when preparing `edge_index` (e.g., add
          reverse edges + self-loops in the data pipeline).
        * No `detach()`/`clone()` trickery: we keep the computation graph intact
          so gradients can flow back to upstream encoders.

    Args:
        in_dim (int): Input feature dimension per node.
        hidden_dim (int) : Hidden feature dimension (also the output dimension).
        num_layers (int, default=2) : Number of stacked GCN layers (>=1).
        dropout (float, default=0.0) : Dropout probability applied after each activation; set to 0.0 to disable.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x, edge_index):
        """
        Run forward GCN inference.

        Args:
            x (torch.Tensor): Node features of shape [N, in_dim] (for the first layer) or
                              [N, hidden_dim] (for subsequent layers).
            edge_index (torch.LongTensor) : COO-style edge index of shape [2, E]. If you need bidirectional
                                            message passing and/or self-loops, build them into this tensor.

        Returns:
            torch.Tensor Encoded node embeddings of shape [N, hidden_dim].
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        return x
