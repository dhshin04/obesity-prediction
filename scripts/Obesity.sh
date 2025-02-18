MODEL='XGBoost'
# MODEL='LogisticRegression'
# MODEL='SVM'

python run.py \
 --model $MODEL \
 --root_path './dataset' \
 --data_path 'obesity_data.csv' \
 --n_trees 30 \
 --max_depth 5 \
 --alpha 0.3 \
 --lambda_ 0.2