#!/bin/bash
# Runs node classification on the given graph

GRAPH="$1"
export PYTHONPATH=".:../:src"
python src/training/train_node_classification.py --graph "${GRAPH}" --task classify --epochs 300