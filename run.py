''' Obesity Prediction using XGBoost '''

from typing import Optional
import os, argparse
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from data_provider.data_loader import ObesityDataset

RANDOM_STATE: Optional[int] = 12


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Loader
    parser.add_argument('--root_path', type=str, required=True, default='./dataset', help='Root path of data file')
    parser.add_argument('--data_path', type=str, required=True, default='obesity_data.csv', help='Data file')

    # Model Define
    parser.add_argument('--model', type=str, required=True, default='XGBoost', help='Model name, options: [XGBoost, LogisticRegression, SVM]')
    parser.add_argument('--n_trees', type=int, default=10, help='Number of trees in ensemble')
    parser.add_argument('--max_depth', type=int, default=5, help='Max depth of decision trees')
    parser.add_argument('--alpha', type=float, default=0.3, help='Learning rate')
    parser.add_argument('--lambda_', type=float, default=0.2, help='L2 regularization hyperparameter')

    args = parser.parse_args()

    # Data Preprocessing
    data_path = os.path.join(os.path.dirname(__file__), args.root_path, args.data_path)
    obesity_dataset = ObesityDataset(data_path, random_state=RANDOM_STATE)

    X_train, X_val, X_test, y_train, y_val, y_test = obesity_dataset.load_data()

    # Model Training
    if args.model == 'XGBoost':
        model = XGBClassifier(
            n_estimators = args.n_trees, 
            max_depth = args.max_depth, 
            learning_rate = args.alpha, 
            reg_lambda = args.lambda_,
            objective = 'multi:softmax',
            num_class=7, 
            random_state=RANDOM_STATE
        )
    elif args.model == 'LogisticRegression':
        model = LogisticRegression(random_state=RANDOM_STATE)
    elif args.model == 'SVM':
        model = SVC()
    else:
        raise Exception('Model is not defined')

    model.fit(X_train, y_train)     # All sklearn and xgboost models follow fit-predict structure

    # Inference
    yhat_val = model.predict(X_val)
    precision_val = precision_score(y_true=y_val, y_pred=yhat_val, average='weighted')
    recall_val = recall_score(y_true=y_val, y_pred=yhat_val, average='weighted')
    f1_val = f1_score(y_true=y_val, y_pred=yhat_val, average='weighted')
    accuracy_val = accuracy_score(y_true=y_val, y_pred=yhat_val)
    print(f'Validation Set Precision: {precision_val*100:.2f}%, Recall: {recall_val*100:.2f}%, F1: {f1_val*100:.2f}%, Accuracy: {accuracy_val*100:.2f}%')

    yhat_test = model.predict(X_test)
    precision_test = precision_score(y_true=y_test, y_pred=yhat_test, average='weighted')
    recall_test = recall_score(y_true=y_test, y_pred=yhat_test, average='weighted')
    f1_test = f1_score(y_true=y_test, y_pred=yhat_test, average='weighted')
    accuracy_test = accuracy_score(y_true=y_test, y_pred=yhat_test)
    print(f'Test Set Precision: {precision_test*100:.2f}%, Recall: {recall_test*100:.2f}%, F1: {f1_test*100:.2f}%, Accuracy: {accuracy_test*100:.2f}%')
