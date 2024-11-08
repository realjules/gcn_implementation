import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.layers import GCNLayer


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super(GCN, self).__init__()
        
        # TODO: add 2 layers of GCN
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.sparse_coo) -> torch.Tensor:
        # given the input node features, and the adjacency matrix, run GCN
        # The order of operations should roughly be:
        # 1. Apply the first GCN layer
        # 2. Apply Relu
        # 3. Apply Dropout
        # 4. Apply the second GCN layer

        # 1. Apply the first GCN layer
        x = self.gcn1(x, adj)
        
        # 2. Apply ReLU activation
        x = F.relu(x)
        
        # 3. Apply Dropout
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 4. Apply the second GCN layer
        output = self.gcn2(x, adj)

        return output
