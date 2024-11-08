@echo off
REM Runs graph classification on the given graph. 

REM To run on MUTAG, set
REM   GRAPH: data/graph_classification/graph_mutag_with_node_num.pt
REM   CLASSES: 2 classes
REM   EPOCHS: 200

REM To run on ENZYMES, set
REM   GRAPH: data/graph_classification/graph_enzymes_with_node_num.pt
REM   CLASSES: 6 classes
REM   EPOCHS: 1000

REM The pooling op can be `mean`, `max`, or `last`.

set GRAPH=%1
set NUM_CLASSES=%2
set POOLING_OP=%3
set NUM_EPOCHS=%4

set PYTHONPATH=.
python src/training/train_graph_classification.py ^
  --task graph_classification ^
  --graphs_path %GRAPH% ^
  --pooling_op %POOLING_OP% ^
  --num_graph_classes %NUM_CLASSES% ^
  --lr 1e-3 ^
  --epochs %NUM_EPOCHS%