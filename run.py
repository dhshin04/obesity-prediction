''' Obesity Prediction '''

from typing import Optional, List
import os, argparse, time
from xgboost import XGBClassifier, plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

from data_provider.data_loader import ObesityDataset
from data_provider.data_factory import data_provider
from models import ObesityNN, node

# For feature importance branch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# For stacking ensemble branch
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from models.custom_log_reg import CustomLogisticRegression
import numpy as np
import pandas as pd


RANDOM_STATE: Optional[int] = 12
RELEVANT_FEATURES: List[str] = [
    'Gender',
    'Age',
    'Height',
    'Weight',
    'family_history',
    'FAVC',
    'FCVC',
    'NCP',
    'CAEC',
    'SMOKE',
    'CH2O',
    'SCC',
    'FAF',
    'TUE',
    'CALC',
    'MTRANS'
]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(args, model, train_loader, val_loader, test_loader):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        start = time.time()
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

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                yhat = model(x_val)
                y_pred = torch.argmax(yhat, dim=1)  # Get class index with max probability
                
                total += y_val.size(0)  # Total number of samples
                correct += (y_pred == y_val).sum().item()  # Count correct predictions
            
            val_accuracy = 100 * correct / total if total > 0 else 0
                
        correct, total = 0, 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
                yhat = model(x_test)

                y_pred = torch.argmax(yhat, dim=1)  # Get class index with max probability
                
                total += y_test.size(0)  # Total number of samples
                correct += (y_pred == y_test).sum().item()  # Count correct predictions
            
            test_accuracy = 100 * correct / total if total > 0 else 0
            
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Time: {time.time() - start:.4f}s')
    
def train_fusion_model(args, X_train, X_val, X_test, y_train, y_val, y_test):
    print("Training fusion model...")
    print(f"Input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=args.n_trees,
            max_depth=args.max_depth,
            learning_rate=args.alpha,
            reg_lambda=args.lambda_,
            objective='multi:softmax',
            num_class=7,
            random_state=RANDOM_STATE
        ),
        'LogisticRegression': LogisticRegression(
            C=args.lambda_,
            max_iter=args.max_iter,
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            C=args.lambda_
        )
    }
    
    for name, model in models.items():
        print(f"\nTraining {name} for fusion...")
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        print(f"{name} training accuracy: {train_score:.4f}")
    
    print("\nTraining Neural Network for fusion...")
    nn_model = ObesityNN.Model(
        input_size=args.n_features,
        output_size=args.n_classes,
        fc1_out=args.fc1_out,
        fc2_out=args.fc2_out,
        fc3_out=args.fc3_out,
        dropout=args.dropout
    )
    nn_model.to(DEVICE)
    
    def get_nn_predictions(model, X):
        print(f"\nGetting NN predictions for {len(X)} samples")
        model.eval()
        
        # Convert to numpy array if it's a DataFrame
        if hasattr(X, 'values'):  # Check if it's a pandas DataFrame
            X_numpy = X.values
        else:
            X_numpy = X  # Assume it's already a numpy array
        
        X_tensor = torch.FloatTensor(X_numpy).to(DEVICE)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        print(f"NN predictions shape: {predicted.shape}")
        return predicted.cpu().numpy()
    
    def add_predictions(X, models, nn_model=None):
        print(f"\nAdding predictions to dataset of shape {X.shape}")
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()
        else:
            # If it's a numpy array, convert to DataFrame
            X_new = pd.DataFrame(X)
            
        for name, model in models.items():
            print(name, model)
            preds = model.predict(X)
            print(X_new, preds)
            print(f"Added {name} predictions - shape: {preds.shape}, unique values: {np.unique(preds)}")
            X_new = X_new.assign(**{f'pred_{name}': preds})
        
        if nn_model:
            nn_preds = get_nn_predictions(nn_model, X)
            print(f"Added NeuralNetwork predictions - shape: {nn_preds.shape}, unique values: {np.unique(nn_preds)}")
            X_new['pred_NeuralNetwork'] = nn_preds
        
        print(f"Final fused dataset shape: {X_new.shape}")
        return X_new
    
    print("\nCreating fused datasets:")
    X_train_fusion = add_predictions(X_train, models, nn_model)
    X_val_fusion = add_predictions(X_val, models, nn_model)
    X_test_fusion = add_predictions(X_test, models, nn_model)
    
    print("\nTraining Custom Logistic Regression on fused data...")
    fusion_model = CustomLogisticRegression(
        learning_rate=args.lr,
        n_iterations=args.max_iter
    )
    
    y_train_binary = (y_train > 3).astype(int)
    print(f"\nBinary class distribution - Train: {np.bincount(y_train_binary)}, Val: {np.bincount((y_val > 3).astype(int))}, Test: {np.bincount((y_test > 3).astype(int))}")
    
    print("\nFitting fusion model...")
    fusion_model.fit(X_train_fusion.values, y_train_binary)
    
    # Evaluate
    def evaluate_fusion(model, X, y_true):
        print(f"\nEvaluating on {len(X)} samples...")
        y_pred = model.predict(X.values)
        print(f"Prediction distribution: {np.bincount(y_pred)}")
        print(f"Actual distribution: {np.bincount(y_true)}")
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        accuracy = accuracy_score(y_true, y_pred)
        return precision, recall, f1, accuracy
    
    y_val_binary = (y_val > 3).astype(int)
    val_precision, val_recall, val_f1, val_accuracy = evaluate_fusion(
        fusion_model, X_val_fusion, y_val_binary
    )
    y_test_binary = (y_test > 3).astype(int)
    test_precision, test_recall, test_f1, test_accuracy = evaluate_fusion(
        fusion_model, X_test_fusion, y_test_binary
    )
    print(f'Validation Set Precision: {val_precision*100:.2f}%, Recall: {val_recall*100:.2f}%, '
        f'F1: {val_f1*100:.2f}%, Accuracy: {val_accuracy*100:.2f}%')

    print(f'Test Set Precision: {test_precision*100:.2f}%, Recall: {test_recall*100:.2f}%, '
        f'F1: {test_f1*100:.2f}%, Accuracy: {test_accuracy*100:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Loader
    parser.add_argument('--root_path', type=str, required=True, default='./dataset', help='Root path of data file')
    parser.add_argument('--data_path', type=str, required=True, default='obesity_data.csv', help='Data file')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of data for validation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for neural network')
    parser.add_argument('--shuffle', type=int, default=1, help='Shuffle flag')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--drop_last', type=int, default=1, help='Drop last incomplete batch for training')
    parser.add_argument('--relevant_features_only', type=int, default=0, help='Only select relevant features')
    parser.add_argument('--doctor', type=int, default=1, help='Select True if you desire the doctor\'s dataset or False if you desire the patient\'s dataset')

    # Model Define
    parser.add_argument('--model', type=str, required=True, default='XGBoost', 
                        help='Model name, options: [XGBoost, LogisticRegression, SVM, NeuralNetwork, NODE, FeatureImportance]')
    
    # XGBoost Hyperparameters
    parser.add_argument('--n_trees', type=int, default=10, help='Number of trees in ensemble')
    parser.add_argument('--max_depth', type=int, default=5, help='Max depth of decision trees')
    parser.add_argument('--alpha', type=float, default=0.3, help='Learning rate')
    parser.add_argument('--lambda_', type=float, default=0.2, help='L2 regularization hyperparameter')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of train iterations')

    # Neural Network Hyperparameters
    parser.add_argument('--n_features', type=int, default=19, help='Number of features in dataset')
    parser.add_argument('--n_classes', type=int, default=7, help='Number of output classes')
    parser.add_argument('--fc1_out', type=int, default=128, help='Neurons in first hidden layer')
    parser.add_argument('--fc2_out', type=int, default=256, help='Neurons in second hidden layer')
    parser.add_argument('--fc3_out', type=int, default=128, help='Neurons in third hidden layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')

    # NODE Hyperparameters
    parser.add_argument('--num_layers', type=int, default=2, help='Number of NODE layers')
    parser.add_argument('--num_trees', type=int, default=32, help='Number of trees per NODE layer')
    parser.add_argument('--tree_dim', type=int, default=8, help='Dimension per tree in NODE model')

    args = parser.parse_args()

    # Data Preprocessing
    data_path_full = os.path.join(os.path.dirname(__file__), args.root_path, args.data_path)
    print(f'Loading data from {data_path_full}...')
    obesity_dataset = ObesityDataset(args, random_state=RANDOM_STATE)

    train_loader, val_loader, test_loader = data_provider(
        args,
        data_path_full,
        args.doctor,
        RELEVANT_FEATURES if args.relevant_features_only else None
    )

    if args.doctor:
        X_train, X_val, X_test, y_train, y_val, y_test = obesity_dataset.load_data_doctor(
            data_path_full, RELEVANT_FEATURES if args.relevant_features_only else None)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = obesity_dataset.load_data_patient(
            data_path_full, RELEVANT_FEATURES if args.relevant_features_only else None)
    

    # Model Training / Feature Importance Identification
    print(f'\nTraining {args.model}...')
    if args.model == 'Fusion':
        print("\n=== Starting Fusion Model Training ===")
        print(f"Original data shapes:")
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
        print(f"Unique classes in y_train: {np.unique(y_train, return_counts=True)}")
        
        train_fusion_model(args, X_train, X_val, X_test, y_train, y_val, y_test)
    elif args.model == 'XGBoost':
        model = XGBClassifier(
            n_estimators = args.n_trees, 
            max_depth = args.max_depth, 
            learning_rate = args.alpha, 
            reg_lambda = args.lambda_,
            objective = 'multi:softmax',
            num_class = 7, 
            random_state = RANDOM_STATE
        )
        model.fit(X_train, y_train)
    elif args.model == 'LogisticRegression':
        model = LogisticRegression(
            C = args.lambda_,
            max_iter = args.max_iter,
            random_state = RANDOM_STATE
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
        train_model(args, model, train_loader, val_loader, test_loader)
    elif args.model == 'NODE':
        model = node.NODEModel(
            input_size=args.n_features,
            output_size=args.n_classes,
            dropout=args.dropout,
            num_layers=args.num_layers,
            num_trees=args.num_trees,
            tree_dim=args.tree_dim
        )
        train_model(args, model, train_loader, val_loader, test_loader)
    elif args.model == 'FeatureImportance':
        print(f'Loading data for feature importance from {data_path_full}...')
        df = pd.read_csv(data_path_full)
        # Define features and target (adjust target name as needed)
        features = RELEVANT_FEATURES
        target = 'Obesity'  # change if your target column is named differently
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found. Available columns: {df.columns.tolist()}")
        # Load a copy of the feature columns
        X = df[features].copy()
        y = df[target]

        # Encode target labels to integers
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Convert all object-type columns to 'category' using apply
        X = X.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)
        
        # Instantiate the XGBoost classifier with enable_categorical=True so categorical types are accepted.
        model = XGBClassifier(
            n_estimators = args.n_trees,
            max_depth = args.max_depth,
            learning_rate = args.alpha,
            reg_lambda = args.lambda_,
            objective = 'multi:softmax',
            num_class = args.n_classes,
            random_state = RANDOM_STATE,
            enable_categorical = True
        )
        print("Training XGBoost model for feature importance...")
        model.fit(X, y)
        
        # Retrieve feature importances based on gain
        importances = model.get_booster().get_score(importance_type='gain')
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        print("Feature Importance (by gain):")
        for feat, imp in sorted_importances:
            print(f"{feat}: {imp:.4g}")
        
        ax = plot_importance(model.get_booster(), importance_type='gain', title='Feature Importance by Gain')
        plt.tight_layout()
        plt.show()
        exit(0)

    elif args.model == 'Stacking':
        print("Training stacking ensemble using data_loader output...")

        # Assume the data loader already returns only the relevant features.
        # Just make sure the data is in DataFrame format.
        if not hasattr(X_train, 'columns'):
            X_train = pd.DataFrame(X_train)
        X_train_subset = X_train.copy()

        if not hasattr(X_val, 'columns'):
            X_val = pd.DataFrame(X_val)
        X_val_subset = X_val.copy()

        if not hasattr(X_test, 'columns'):
            X_test = pd.DataFrame(X_test)
        X_test_subset = X_test.copy()

        # Convert any object-type columns to numeric codes.
        def to_numeric(df):
            df_numeric = df.copy()
            for col in df_numeric.columns:
                if df_numeric[col].dtype == 'object':
                    df_numeric[col] = df_numeric[col].astype('category').cat.codes
            return df_numeric

        X_train_stack = to_numeric(X_train_subset.copy())
        X_val_stack   = to_numeric(X_val_subset.copy())
        X_test_stack  = to_numeric(X_test_subset.copy())

        # Define base estimators.

        base_estimators = [
            ('xgb', XGBClassifier(
                n_estimators=args.n_trees,
                max_depth=args.max_depth,
                learning_rate=args.alpha,
                reg_lambda=args.lambda_,
                objective='multi:softmax',
                num_class=args.n_classes,
                random_state=RANDOM_STATE,
                enable_categorical=True
            )),
            ('lr', LogisticRegression(max_iter=args.max_iter, random_state=RANDOM_STATE)),
            ('svm', SVC(probability=True, C=args.lambda_, random_state=RANDOM_STATE))
        ]
        
        # Use RandomForestClassifier as the final estimator.
        final_estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=5,
            passthrough=False
        )
        
        print("Training stacking ensemble...")
        stacking_model.fit(X_train_stack, y_train)
        model = stacking_model
        # Override inference data with the numeric versions.
        X_val = X_val_stack
        X_test = X_test_stack

    else:
        raise Exception('Model is not defined')

    if args.model not in ['NeuralNetwork', 'FeatureImportance', 'Fusion']:
        # Inference for models that implement predict()
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
    