import torch
import torch.nn as nn


class GraphClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, pooling_op: str):
        super(GraphClassifier, self).__init__()
        print(f"Using pooling operation: {pooling_op}")
        
        self.pooling_op = pooling_op
        
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given node features x, applies two operations:
        1. Pools the node representations using the pooling operation specified
        2. Applies a classifier to the pooled representation
        """
        # Get node features and apply pooling
        pooled = self.pool(x, self.pooling_op)
        
        # Apply classifier
        logits = self.graph_classifier(pooled)

        return logits.squeeze(0)  # Remove extra dimension to get [batch_size, num_classes]
    

    def pool(self, x: torch.Tensor, operation: str = "last") -> torch.Tensor:
        """Given node features x, applies a pooling operation to return a 
        single aggregated feature vector.

        Args:
            x (torch.Tensor): [The node features]
            operation (str, optional): [description]. Defaults to "last".

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: [A single feature vector for the graph]
        """
        if operation == "mean":
            return x.mean(dim=0)
        elif operation == "max":
            return x.max(dim=0)[0]
        elif operation == "last":
            return x[-1]
        else: # Adding a detailed error note to help with debugging.
            raise NotImplementedError(f"Pooling operation {operation} not implemented")