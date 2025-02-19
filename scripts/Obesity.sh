python run.py \
 --model 'LogisticRegression' \
 --root_path './dataset' \
 --data_path 'obesity_data.csv' \
 --lambda_ 1000 \
 --max_iter 200

python run.py \
 --model 'XGBoost' \
 --root_path './dataset' \
 --data_path 'obesity_data.csv' \
 --n_trees 30 \
 --max_depth 5 \
 --alpha 0.3 \
 --lambda_ 0.2

python run.py \
 --model 'SVM' \
 --root_path './dataset' \
 --data_path 'obesity_data.csv' \
 --lambda_ 100