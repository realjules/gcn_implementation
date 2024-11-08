@echo off
set PYTHONPATH=.
python src/training/train_node_classification.py --graph %1 --task classify --epochs 300