import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.layers import GCNLayer

class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        """GCN with variable number of layers
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(GCN, self).__init__()
        
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Last layer: hidden_dim -> output_dim
        self.layers.append(GCNLayer(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, adj: torch.sparse_coo) -> torch.Tensor:
        # Apply dropout to input
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Pass through all layers except last
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Last layer without ReLU
        x = self.layers[-1](x, adj)
        return x