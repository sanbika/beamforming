import torch
from torch import nn


class NNw(torch.nn.Module):
    def __init__(self, Tx, Rx, user_num, hidden_nodes):
        super(NNw, self).__init__()
        self.linear1 = nn.Linear(2 * Tx * Rx * user_num, hidden_nodes)
        self.act1 = nn.ReLU()
        self.linear2= nn.Linear(hidden_nodes, user_num)

    def forward(self, x):
        x = x.reshape(1, -1)
        out = self.linear1(x)
        out = self.act1(out)
        out = self.linear2(out)
        return out


class NNu(torch.nn.Module):
    def __init__(self, Tx, Rx, user_num, hidden_nodes):
        super(NNu, self).__init__()
        self.linear1 = nn.Linear(2 * Tx * Rx * user_num, hidden_nodes)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_nodes, 2 * Rx * user_num)


    def forward(self, x):
        x = x.reshape(1, -1)
        out = self.linear1(x)
        out = self.act1(out)
        out = self.linear2(out)
        return out


class NNmu(torch.nn.Module):
    def __init__(self, Tx, Rx, user_num, hidden_nodes):
        super(NNmu, self).__init__()
        self.linear1 = nn.Linear(2 * Tx * Rx * user_num, hidden_nodes)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        x = x.reshape(1, -1)
        out = self.linear1(x)
        out = self.act1(out)
        out = self.linear2(out)
        return out