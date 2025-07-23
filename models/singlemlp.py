import torch 
import torch.nn as nn


class SingleMLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, output_size):
        super(SingleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, output_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        x = self.softplus(x)
        return x