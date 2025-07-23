import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.input_size = self.args.seq_len
        self.output_size = self.args.pred_len
        self.hidden_size_1 = self.args.hidden_dim
        self.hidden_size_2 = 128
        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x, dec_inp):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x.reshape(x.size(0), self.output_size, -1)
        return x