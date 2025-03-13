from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


def preprocess(data_path: str, relevant_features: List[str] = None) -> Tuple:
    df = pd.read_csv(data_path)

    # Data Cleaning
    df = df.dropna(axis=0, how='any')

    features = df.drop(columns=['Obesity'])
    target = df['Obesity']

    binary_map = {'yes': 1, 'no': 0}
    features['family_history'] = features['family_history'].map(binary_map)
    features['FAVC'] = features['FAVC'].map(binary_map)
    features['SMOKE'] = features['SMOKE'].map(binary_map)
    features['SCC'] = features['SCC'].map(binary_map)

    # Convert ordinal categorical variables using Label Encoding
    ordinal_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    features['CAEC'] = features['CAEC'].map(ordinal_map)
    features['CALC'] = features['CALC'].map(ordinal_map)

    # Ensure numerical columns are of the correct type
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    features[num_cols] = features[num_cols].astype(float)

    # Filter by relevant features
    if relevant_features:
        features = features[relevant_features]

    # Apply One-Hot Encoding to nominal categorical variables
    columns = ['Gender', 'MTRANS']
    if relevant_features:
        columns = [col for col in columns if col in relevant_features]
    if len(columns) > 0:
        features = pd.get_dummies(features, columns=columns, drop_first=True)  # drop_first=True to avoid dummy variable trap

    # Define manual encoding for the Obesity column
    obesity_mapping = {
        "Insufficient_Weight": 0, 
        "Normal_Weight": 1,
        "Overweight_Level_I": 2,
        "Overweight_Level_II": 3,
        "Obesity_Type_I": 4,  
        "Obesity_Type_II": 5, 
        "Obesity_Type_III": 6
    }

    # Apply manual encoding
    target = target.map(obesity_mapping)

    features = features.to_numpy()
    target = target.to_numpy()

    return features, target


class ObesityDataset():
    def __init__(self, args, Scaler=None, random_state=None):
        self.scaler = Scaler() if Scaler else StandardScaler()
        self.random_state = random_state
        self.val_test_ratio = 1 - args.train_ratio
        self.test_ratio = 1 - (args.val_ratio / self.val_test_ratio)

    def load_data(self, data_path: str, relevant_features: List[str] = None) -> Tuple:
        # Data Preprocessing
        X, y = preprocess(data_path, relevant_features)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.val_test_ratio, random_state=self.random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=self.test_ratio, random_state=self.random_state)

        # Normalize Data
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test


class ObesityTorchDataset(Dataset):
    def __init__(self, data_path: str, relevant_features: List[str] = None, Scaler=None):
        self.features, self.target = preprocess(data_path, relevant_features)

        # Normalize Data
        scaler = Scaler() if Scaler else StandardScaler()
        self.features = scaler.fit_transform(self.features)

        # Convert to tensor
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.long)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

    def __len__(self):
        if self.features.size(0) != self.target.size(0):
            raise Exception('Sample size mismatch between features and target')
        return self.features.size(0)
