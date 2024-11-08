import logging
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from src.data.utils import seed_all, accuracy
from src.modeling.tasks.node_classification_layers import NodeClassifier
from src.training.args import get_training_args
from src.data.graph import Graph
from src.data.constants import *

class LayerAnalysisTrainer(object):
    def __init__(self, args, num_layers) -> None:
        super().__init__()
        self.args = args
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        
        # Load graph
        self.graph = Graph(**vars(self.args))
        
        # Initialize model with specified number of layers
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
            num_layers=self.num_layers  # Pass number of layers to model
        ).to(self.device)
        
    def train_and_eval(self):
        best_val_acc = 0
        final_metrics = {}
        
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
                final_metrics = {
                    'train_loss': loss_train.item(),
                    'train_acc': acc_train.item(),
                    'val_loss': loss_val.item(),
                    'val_acc': acc_val.item()
                }
            
            if (epoch + 1) % 100 == 0:
                print(f'Layer {self.num_layers}, Epoch {epoch+1:04d}:', end=' ')
                print(f'loss_train: {loss_train.item():.4f},', end=' ')
                print(f'acc_train: {acc_train.item():.4f},', end=' ')
                print(f'loss_val: {loss_val.item():.4f},', end=' ')
                print(f'acc_val: {acc_val.item():.4f}')
        
        return final_metrics

def plot_layer_analysis(results, dataset_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    layers = list(results.keys())
    train_loss = [results[l]['train_loss'] for l in layers]
    val_loss = [results[l]['val_loss'] for l in layers]
    train_acc = [results[l]['train_acc'] for l in layers]
    val_acc = [results[l]['val_acc'] for l in layers]
    
    # Plot losses
    ax1.plot(layers, train_loss, 'b-o', label='Train Loss')
    ax1.plot(layers, val_loss, 'r-o', label='Val Loss')
    ax1.set_xlabel('Number of Layers')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{dataset_name} - Loss vs Number of Layers')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(layers, train_acc, 'b-o', label='Train Accuracy')
    ax2.plot(layers, val_acc, 'r-o', label='Val Accuracy')
    ax2.set_xlabel('Number of Layers')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{dataset_name} - Accuracy vs Number of Layers')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'layer_analysis_{dataset_name.lower()}.png')
    print(f"Plot saved as layer_analysis_{dataset_name.lower()}.png")

def main():
    # Get base arguments
    args = get_training_args()
    seed_all(args.seed)
    
    # Layer configurations to test
    layer_configs = [3, 4, 5, 6]
    
    # Store results for each layer configuration
    results = {}
    
    for num_layers in layer_configs:
        print(f"\nTraining with {num_layers} layers...")
        trainer = LayerAnalysisTrainer(args, num_layers)
        metrics = trainer.train_and_eval()
        results[num_layers] = metrics
    
    # Print summary
    print("\nSummary of results:")
    print("Layers | Train Loss | Val Loss | Train Acc | Val Acc")
    print("-" * 50)
    for layers in layer_configs:
        r = results[layers]
        print(f"{layers:6d} | {r['train_loss']:9.4f} | {r['val_loss']:8.4f} | "
              f"{r['train_acc']:8.4f} | {r['val_acc']:7.4f}")
    
    # Plot results
    plot_layer_analysis(results, args.graph.upper())

if __name__ == "__main__":
    main()