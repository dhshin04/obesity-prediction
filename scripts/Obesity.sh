# TODO: Requires hyperparameter tuning

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

python run.py \
 --model 'NeuralNetwork' \
 --root_path './dataset' \
 --data_path 'obesity_data.csv' \
 --train_ratio 0.7 \
 --val_ratio 0.15 \
 --batch_size 16 \
 --shuffle 1 \
 --num_workers 1 \
 --drop_last 1 \
 --n_features 19 \
 --n_classes 7 \
 --fc1_out 128 \
 --fc2_out 256 \
 --fc3_out 128 \
 --dropout 0.2 \
 --lr 0.001 \
 --epochs 20