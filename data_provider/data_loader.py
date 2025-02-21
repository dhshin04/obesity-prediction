from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


def preprocess(data_path: str) -> Tuple:
    df = pd.read_csv(data_path)

    # Data Cleaning
    df = df.dropna(axis=0, how='any')

    binary_map = {'yes': 1, 'no': 0}
    df['family_history'] = df['family_history'].map(binary_map)
    df['FAVC'] = df['FAVC'].map(binary_map)
    df['SMOKE'] = df['SMOKE'].map(binary_map)
    df['SCC'] = df['SCC'].map(binary_map)

    # Convert ordinal categorical variables using Label Encoding
    ordinal_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    df['CAEC'] = df['CAEC'].map(ordinal_map)
    df['CALC'] = df['CALC'].map(ordinal_map)

    # Apply One-Hot Encoding to nominal categorical variables
    df = pd.get_dummies(df, columns=['Gender', 'MTRANS'], drop_first=True)  # drop_first=True to avoid dummy variable trap

    # Ensure numerical columns are of the correct type
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    df[num_cols] = df[num_cols].astype(float)

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
    df['Obesity'] = df['Obesity'].map(obesity_mapping)

    target = df['Obesity'].to_numpy()
    features = df.drop(columns=['Obesity']).to_numpy()

    return features, target


class ObesityDataset():
    def __init__(self, Scaler=None, random_state=None):
        self.scaler = Scaler() if Scaler else StandardScaler()
        self.random_state = random_state

    def load_data(self, data_path: str) -> Tuple:
        # Data Preprocessing
        X, y = preprocess(data_path)

        # Split Data
        # TODO: Change hard-coded test size with args
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=self.random_state)

        # Normalize Data
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test


class ObesityTorchDataset(Dataset):
    def __init__(self, data_path: str, Scaler=None):
        self.features, self.target = preprocess(data_path)

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
