import torch 
import torch.nn as nn 


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.input_dim = self.args.seq_len
        self.feature_size = self.args.feature_size
        self.output_dim = self.args.pred_len
        self.hidden_dim = self.args.hidden_dim
        self.layer_dim = self.args.layer_dim
        self.gru = nn.GRU(self.feature_size, self.hidden_dim, self.layer_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        # self.relu = nn.ReLU()
        # self.softplus = nn.Softplus()
        # self.leakrelu = nn.LeakyReLU()
    
    def forward(self, x, dec_inp):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        out = out.reshape(-1, self.output_dim, self.feature_size)
        # out = self.relu(out)
        # out = self.softplus(out)
        return out
