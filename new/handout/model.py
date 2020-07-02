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
        y = torch.zeros((x.shape[0], 128), dtype=torch.float64).cuda()
        for i in range(x.shape[0]):
            s = x[i]
            s = [[[float(ord(s[i]))] for i in range(len(s))]]
            s = torch.tensor(s, dtype=torch.float64).cuda()
            rs, h = self.lstm(s)
#            print(rs.size())
            y[i] =  rs[0][len(x[i]) - 1][:]
        y = self.line1(y).view(x.shape[0])
        return y

    def predict(self, x):
        n = x.shape[0]
        i = 0
        y = torch.zeros(0).double().cuda()
        while i < n:
            j = min(i + 10, n)
            p = self.forward(x[i: j])
            y = torch.cat((y, p))
            i = j
        return y.cpu()

class TreeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=4, batch_first=True, bidirectional=True)
        self.line = nn.Linear(128, 1)
        self.conv = nn.Conv1d(128, 128)
    
    def forward(self, x):
        y = torch.zeros((x.shape[0], 128), dtype=torch.float64).cuda()
        for i in range(x.shape[0]):
            s = x[i]
            s = [[[float(ord(s[i]))] for i in range(len(s))]]
            s = torch.tensor(s, dtype=torch.float64).cuda()
            rs, h = self.lstm(s)
#            print(rs.size())
            y[i] =  rs[0][len(x[i]) - 1][:]
        y = self.line1(y).view(x.shape[0])
        return y

    def predict(self, x):
        n = x.shape[0]
        i = 0
        y = torch.zeros(0).double().cuda()
        while i < n:
            j = min(i + 10, n)
            p = self.forward(x[i: j])
            y = torch.cat((y, p))
            i = j
        return y.cpu()