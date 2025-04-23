#!/bin/bash
# This script runs the feature importance identification using run.py.

python3 run.py \
    --root_path "./dataset" \
    --data_path "obesity_dataset_patient.csv" \
    --model FeatureImportance \
    --n_trees 10 \
    --max_depth 5 \
    --alpha 0.3 \
    --lambda_ 0.2 \
    --n_features 14 \
    --n_classes 7 \
    --doctor 0