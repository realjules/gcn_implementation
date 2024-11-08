import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.gcn import GCN


class LinkPrediction(nn.Module):
    def __init__(self, hidden_dim: int):
        """Link prediction module.
        We want to predict the edge label (0 or 1) for each edge.
        We assume that the model gets the node features as input (i.e., GCN is already applied to the node features).
        Args:
            hidden_dim (int): [The hidden dimension of the GCN layer (i.e., feature dimension of the nodes)]
        """

        super(LinkPrediction, self).__init__()
        
        # The edge classifier takes concatenated node features (2 * hidden_dim)
        # and outputs 2 logits (one for each class - edge exists or not)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output 2 logits: [no_edge, edge]
        )
        
    def forward(
        self, node_features_after_gcn: torch.Tensor, edges: torch.Tensor,
    ) -> torch.Tensor:

        # node_features_after_gcn: [num_nodes, hidden_dim]
        # edges: [2, num_edges]
        # the function should return classifier logits for each edge
        # Note that the output should not be probabilities, rather one logit for each class (so the output should be batch_size x 2).
        # TODO: Implement the forward pass of the link prediction module
        
        # 1. Get embeddings for both nodes of each edge
        node1_features = node_features_after_gcn[edges[0]]  # [num_edges, hidden_dim]
        node2_features = node_features_after_gcn[edges[1]]  # [num_edges, hidden_dim]
        
        # 2. Concatenate features for each edge
        edge_features = torch.cat([node1_features, node2_features], dim=1)  # [num_edges, 2*hidden_dim]
        
        # 3. Apply classifier to get logits
        classifier_logits = self.edge_classifier(edge_features)  # [num_edges, 2]
        
        return classifier_logits

