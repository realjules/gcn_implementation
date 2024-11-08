import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.gcn import GCN


class NodeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        """The node classifier module.
            we need two components for the node classification task:
            1. a GCN model, that can learn informative features for each node
            2. a linear classifier, that takes the features for each node and uses them to predict the node label
        Args:
            input_dim (int): [Input feature dimensions (i.e. the number of features for each node)]
            hidden_dim (int): [Hidden dimensions for the GCN model]
            n_classes (int): [Number of classes for the node classification task]
            dropout (float): [Dropout rate for the GCN model]
        """
        super(NodeClassifier, self).__init__()

        self.gcn = GCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # GCN outputs features of hidden_dim size
            dropout=dropout
        )
        
        # Initialize the linear classifier
        # Takes hidden_dim features and outputs n_classes scores
        self.node_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(
        self, x: torch.Tensor, adj: torch.sparse_coo, classify: bool = True
    ) -> torch.Tensor:
        # TODO: implement the forward pass of the node classification task

        # 1. Get node embeddings from GCN
        node_embeddings = self.gcn(x, adj)
        
        # 2. If classify=False, return the embeddings
        if not classify:
            return node_embeddings
        
        # 3. Apply the classifier to get class logits
        logits = self.node_classifier(node_embeddings)
        
        return F.log_softmax(logits, dim=1)
        
