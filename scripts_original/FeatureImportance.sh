#!/bin/bash
# filepath: c:\Users\kutha\OneDrive\Documents\CS 4774 Machine Learning\Project\obesity-prediction\scripts\feature_importance.sh
# This script runs the feature importance identification using run.py.

python3 ../run.py \
    --root_path "./dataset" \
    --data_path "obesity_data.csv" \
    --model FeatureImportance \
    --n_trees 10 \
    --max_depth 5 \
    --alpha 0.3 \
    --lambda_ 0.2 \
    --n_features 19 \
    --n_classes 7