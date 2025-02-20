''' Obesity Prediction using XGBoost '''

from typing import Optional
import os, argparse
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

from data_provider.data_loader import ObesityDataset
from data_provider.data_factory import data_provider
from models import ObesityNN

RANDOM_STATE: Optional[int] = 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(args, model, train_loader, val_loader, test_loader):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)

            optimizer.zero_grad()
            yhat = model(x_train)
            loss = criterion(yhat, y_train)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # TODO: Add accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                yhat = model(x_val)
        
        # TODO: Add accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
                yhat = model(x_test)
            
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Loader
    parser.add_argument('--root_path', type=str, required=True, default='./dataset', help='Root path of data file')
    parser.add_argument('--data_path', type=str, required=True, default='obesity_data.csv', help='Data file')

    # Model Define
    parser.add_argument('--model', type=str, required=True, default='XGBoost', help='Model name, options: [XGBoost, LogisticRegression, SVM, NeuralNetwork]')
    parser.add_argument('--n_trees', type=int, default=10, help='Number of trees in ensemble')
    parser.add_argument('--max_depth', type=int, default=5, help='Max depth of decision trees')
    parser.add_argument('--alpha', type=float, default=0.3, help='Learning rate')
    parser.add_argument('--lambda_', type=float, default=0.2, help='L2 regularization hyperparameter')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of train iterations')

    args = parser.parse_args()

    # Data Preprocessing
    data_path = os.path.join(os.path.dirname(__file__), args.root_path, args.data_path)
    obesity_dataset = ObesityDataset(random_state=RANDOM_STATE)

    train_loader, val_loader, test_loader = data_provider(
        args,
        data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    X_train, X_val, X_test, y_train, y_val, y_test = obesity_dataset.load_data(data_path)

    # Model Training
    print(f'\nTraining {args.model}...')
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
        model.fit(X_train, y_train)
    elif args.model == 'LogisticRegression':
        model = LogisticRegression(
            C = args.lambda_,
            max_iter = args.max_iter,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
    elif args.model == 'SVM':
        model = SVC(
            C = args.lambda_
        )
        model.fit(X_train, y_train)
    elif args.model == 'NeuralNetwork':
        model = ObesityNN.Model(
            input_size=args.n_features,
            output_size=args.n_classes,
            fc1_out=args.fc1_out,
            fc2_out=args.fc2_out,
            fc3_out=args.fc3_out,
            dropout=args.dropout
        )
        train_model(args, model, train_loader, val_loader)
    else:
        raise Exception('Model is not defined')

    if args.model != 'NeuralNetwork':
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
