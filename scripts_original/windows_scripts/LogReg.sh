#!/bin/bash
python3 -u ../../run.py \
 --model 'LogisticRegression' \
 --root_path './dataset' \
 --data_path 'obesity_data.csv' \
 --relevant_features_only 0 \
 --lambda_ 1000 \
 --max_iter 200