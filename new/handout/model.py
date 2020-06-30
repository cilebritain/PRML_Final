import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=4, batch_first=True, bidirectional=True)
        self.line1 = nn.Linear(128, 1)
    
    def forward(self, x):
        rs, h = self.lstm(x)
        rs = self.line1(rs) 
        rs = rs.view([rs.size()[0], rs.size()[1]])
        y = torch.tensor(np.zeros(rs.size()[0]))
        for i in range(rs.size()[0]):
            y[i] = max(rs[i])
        return y

    def predict(self, x):
        n = x.size()[0]
        i = 0
        y = torch.zeros(0).double()
        while i < n:
            j = min(i + 10, n)
            p = self.forward(x[i: j])
            y = torch.cat((y, p))
            i = j
        return y
    