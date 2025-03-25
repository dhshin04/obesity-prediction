#!/bin/bash
python3 -u ../../run.py \
 --model 'LogisticRegression' \
 --root_path './dataset' \
 --data_path 'obesity_dataset_patient.csv' \
 --relevant_features_only 0 \
 --lambda_ 1000 \
 --max_iter 200 \
 --doctor 0