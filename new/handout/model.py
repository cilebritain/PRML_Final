import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, bidirectional=True)
        self.line1 = nn.Linear(64, 2)
    
    def forward(self, x):
        y = torch.zeros((x.shape[0], 64), dtype=torch.float64).cuda()
        for i in range(x.shape[0]):
            s = x[i]
            s = [[[float(ord(s[i]))] for i in range(len(s))]]
            s = torch.tensor(s, dtype=torch.float64).cuda()
            rs, h = self.lstm(s)
#            print(rs.size())
            y[i] = rs[0][len(x[i]) - 1][:]
        y = self.line1(y)
        return y

    def predict(self, x):
        n = x.shape[0]
        i = 0
        y = []
        for i in range(n):
            p = self.forward(x[i: i + 1])
            y.append(torch.argmax(p[0]))
        return torch.tensor(y)


class TreeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(64, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.line = nn.Linear(64, 2)
    
    def forward(self, s, c):
        y = torch.zeros(0, dtype=torch.float64).cuda()
        for i in range(len(c)):
            m = len(c[i])
            f = [torch.zeros(64) for j in range (m)]
            for j in range(m - 1, -1 , -1):
                cc = 0
                a = []
                sub = []
                for k in range(len(s[i][j])):
                    if cc == len(c[i][j]) or k != c[i][j][cc][1]: 
                        sub.append([ord(s[i][j][k])])
                    else:
                        if len(sub) != 0:
                            ts = torch.tensor([sub], dtype=torch.float64).cuda()
                            rs, h = self.lstm1(ts)
                            a.append(rs[0][len(sub) - 1][:])
                            sub = []
                        a.append(f[c[i][j][cc][0]])

                if len(sub) != 0:
                    ts = torch.tensor([sub], dtype=torch.float64).cuda()
                    rs, h = self.lstm1(ts)
                    a.append(rs[0][len(sub) - 1][:])

                ts = torch.zeros(0, dtype=torch.float64).cuda()
                for x in a:
                    ts = torch.cat((ts, x), 0)
                ts = ts.view([1, len(a), a[0].size()[0]])
                rs, h = self.lstm2(ts)
                f[j] = rs[0][len(a) - 1][:]
            
            rs = self.line(f[0])
            rs = torch.softmax(rs, 0)
            y = torch.cat((y, rs.view([1, 2])), 0)

        return y

    def predict(self, s, c):
        n = len(s)
        i = 0
        y = []
        for i in range(n):
            p = self.forward(s[i: i + 1], c[i: i + 1])
            y.append(torch.argmax(p[0]))

        return torch.tensor(y)