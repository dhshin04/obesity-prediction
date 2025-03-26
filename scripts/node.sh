#!/bin/bash
# This script runs the NODE model using run.py with preset parameters.
python3 ../run.py \
    --root_path './dataset' \
    --data_path 'obesity_data.csv' \
    --model NODE \
    --relevant_features_only 0 \
    --n_features 19 \
    --n_classes 7 \
    --dropout 0.2 \
    --lr 0.001 \
    --epochs 10 \
    --num_layers 2 \
    --num_trees 32 \
    --tree_dim 8