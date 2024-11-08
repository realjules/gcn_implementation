import logging
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tabulate import tabulate

from src.data.utils import seed_all, accuracy
from src.modeling.tasks.node_classification import NodeClassifier
from src.training.args import get_training_args
from src.data.graph import Graph
from src.data.constants import *

class FeatureAnalysisTrainer(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        
        # Load graph
        self.graph = Graph(**vars(self.args))
        
        self.init_model()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def init_model(self):
        self.model = NodeClassifier(
            input_dim=self.graph.features.shape[1],
            hidden_dim=self.args.hidden,
            n_classes=self.graph.labels.max().item() + 1,
            dropout=self.args.dropout,
        ).to(self.device)
        
    def train_and_eval(self):
        best_val_acc = 0
        best_metrics = {}
        
        t_total = time.time()
        for epoch in range(self.args.epochs):
            # Training
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            
            output = self.model(self.graph.features, self.graph.adj)
            loss_train = F.nll_loss(output[self.graph.idx_train], self.graph.labels[self.graph.idx_train])
            acc_train = accuracy(output[self.graph.idx_train], self.graph.labels[self.graph.idx_train])
            
            loss_train.backward()
            self.optimizer.step()

            # Validation
            self.model.eval()
            output = self.model(self.graph.features, self.graph.adj)
            loss_val = F.nll_loss(output[self.graph.idx_val], self.graph.labels[self.graph.idx_val])
            acc_val = accuracy(output[self.graph.idx_val], self.graph.labels[self.graph.idx_val])
            
            # Save best validation accuracy
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                
                # Test metrics at best validation
                loss_test = F.nll_loss(output[self.graph.idx_test], self.graph.labels[self.graph.idx_test])
                acc_test = accuracy(output[self.graph.idx_test], self.graph.labels[self.graph.idx_test])
                
                best_metrics = {
                    'train_loss': loss_train.item(),
                    'train_acc': acc_train.item(),
                    'val_loss': loss_val.item(),
                    'val_acc': acc_val.item(),
                    'test_loss': loss_test.item(),
                    'test_acc': acc_test.item()
                }
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1:04d}:', end=' ')
                print(f'loss_train: {loss_train.item():.4f},', end=' ')
                print(f'acc_train: {acc_train.item():.4f},', end=' ')
                print(f'loss_val: {loss_val.item():.4f},', end=' ')
                print(f'acc_val: {acc_val.item():.4f}')
        
        return best_metrics

def main():
    # Get base arguments
    args = get_training_args()
    seed_all(args.seed)
    
    datasets = ['cora', 'citeseer']
    feature_types = ['', '_topo', '_plus_topo']
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        
        for feature_type in feature_types:
            print(f"\nTraining {dataset}{feature_type}")
            
            # Update graph name in args
            args.graph = f"{dataset}{feature_type}"
            
            # Train and evaluate
            trainer = FeatureAnalysisTrainer(args)
            metrics = trainer.train_and_eval()
            results[dataset][feature_type] = metrics
    
    # Print results in a nice table
    print("\nResults Summary:")
    headers = ["Dataset", "Feature Type", "Train Acc", "Val Acc", "Test Acc"]
    table_data = []
    
    for dataset in datasets:
        for feature_type in feature_types:
            r = results[dataset][feature_type]
            feature_name = "Original" if feature_type == "" else (
                "Topological" if feature_type == "_topo" else "Topological Plus"
            )
            table_data.append([
                dataset.upper(),
                feature_name,
                f"{r['train_acc']*100:.2f}%",
                f"{r['val_acc']*100:.2f}%",
                f"{r['test_acc']*100:.2f}%"
            ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()