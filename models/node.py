import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NODELayer(nn.Module):
    def __init__(self, hidden_dim, num_trees, tree_dim, dropout=0.2, bn=True):
        """
        Initializes a single NODELayer.
        
        Args:
            hidden_dim (int): The input (and output) dimension, computed as num_trees * tree_dim.
            num_trees (int): Number of trees (i.e., decision ensembles) in this layer.
            tree_dim (int): Dimension per tree (number of leaves per tree).
            dropout (float): Dropout rate for regularization.
            bn (bool): Whether to apply batch normalization.
        """
        super().__init__()
        self.num_trees = num_trees
        self.tree_dim = tree_dim
        # Linear layer to compute decision gates for each tree.
        self.decision = nn.Linear(hidden_dim, num_trees)
        # Linear layer to compute leaf outputs for each tree (flattened).
        self.leaf = nn.Linear(hidden_dim, num_trees * tree_dim)
        # Dropout for regularization.
        self.dropout = nn.Dropout(dropout)
        # Optional batch normalization after dropout.
        self.bn = nn.BatchNorm1d(num_trees * tree_dim) if bn else None

    def forward(self, x):
        """
        Forward pass for the NODELayer.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, hidden_dim].
        
        Returns:
            Tensor: Output tensor of shape [batch_size, num_trees * tree_dim].
        """
        # Compute decision gates with sigmoid activation.
        decisions = torch.sigmoid(self.decision(x))  # shape: [batch_size, num_trees]
        # Compute leaf outputs and reshape to [batch_size, num_trees, tree_dim].
        leaf_out = self.leaf(x).view(x.size(0), self.num_trees, self.tree_dim)
        # Multiply leaf outputs by expanded decision gates and flatten.
        out = (leaf_out * decisions.unsqueeze(-1)).view(x.size(0), -1)
        # Apply dropout.
        out = self.dropout(out)
        # Optionally apply batch normalization.
        if self.bn:
            out = self.bn(out)
        return out

class NODEModel(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2, num_layers=3, num_trees=32, tree_dim=8, bn=True):
        """
        Initializes the NODE model with increased complexity using residual connections and an extra hidden layer.
        
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output classes.
            dropout (float): Dropout rate for NODE layers.
            num_layers (int): Number of NODE layers to stack.
            num_trees (int): Number of trees in each NODE layer.
            tree_dim (int): Dimension per tree (defines hidden_dim as num_trees * tree_dim).
            bn (bool): Whether to use Batch Normalization.
        """
        super().__init__()
        # Hidden dimension computed as the product of num_trees and tree_dim.
        self.hidden_dim = num_trees * tree_dim
        # Project input data to the hidden dimension.
        self.input_proj = nn.Linear(input_size, self.hidden_dim)
        # Optional batch normalization for the input projection.
        self.input_bn = nn.BatchNorm1d(self.hidden_dim) if bn else None
        # Create a list of NODE layers to allow residual connections.
        self.layers = nn.ModuleList([
            NODELayer(self.hidden_dim, num_trees, tree_dim, dropout, bn)
            for _ in range(num_layers)
        ])
        # An extra fully-connected layer to increase model capacity.
        self.extra_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Final classifier layer that produces logits for the output classes.
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x):
        """
        Forward pass for the NODE model.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, input_size].
        
        Returns:
            Tensor: Logits tensor of shape [batch_size, output_size].
        """
        # Project input to hidden dimension.
        x = self.input_proj(x)
        if self.input_bn:
            x = self.input_bn(x)
        # Apply ReLU activation.
        x = F.relu(x)
        # Process through each NODE layer with a residual connection.
        for layer in self.layers:
            residual = x  # Save input for the residual connection.
            x = layer(x)
            x = x + residual  # Add the residual connection.
            x = F.relu(x)     # Apply activation after the addition.
        # Pass through an extra fully-connected layer for additional complexity.
        x = F.relu(self.extra_fc(x))
        # Final classification layer.
        return self.fc(x)
    
    def predict(self, X):
        """
        Predicts class indices for input data.
        
        Args:
            X (Tensor or np.ndarray): Input data.
        
        Returns:
            Tensor: Predicted class indices.
        """
        self.eval()  # Set model to evaluation mode.
        # Ensure X is a torch.Tensor.
        if not torch.is_tensor(X):
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            else:
                X = torch.tensor(X, dtype=torch.float)
        # Disable gradient computation for evaluation.
        with torch.no_grad():
            outputs = self.forward(X)
            preds = torch.argmax(outputs, dim=1)
        return preds