import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_size, output_size, fc1_out=128, fc2_out=256, fc3_out=128, dropout=0.0):
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, fc3_out)
        self.output = nn.Linear(fc3_out, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.output(x)
        return x
