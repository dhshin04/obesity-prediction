#!/bin/bash

python3 run.py \
  --root_path './dataset' \
  --data_path 'obesity_dataset_patient.csv' \
  --model Stacking \
  --n_trees 100 \
  --max_depth 6 \
  --alpha 0.1 \
  --lambda_ 1.0 \
  --n_features 14 \
  --n_classes 7 \
  --fc1_out 128 \
  --fc2_out 256 \
  --fc3_out 128 \
  --dropout 0.2 \
  --lr 0.001 \
  --epochs 20 \
  --max_iter 1000 \
  --doctor 0