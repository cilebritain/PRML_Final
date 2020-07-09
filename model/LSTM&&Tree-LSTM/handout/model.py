import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, bidirectional=True)
        self.line1 = nn.Linear(64, 2)
    
    def forward(self, x):
        y = Variable(torch.zeros((x.shape[0], 64), dtype=torch.float64), requires_grad=True).cuda()
        for i in range(x.shape[0]):
            s = x[i]
            s = [[[float(ord(s[i]))] for i in range(len(s))]]
            s = torch.tensor(s, dtype=torch.float64, requires_grad=True).cuda()
            rs, h = self.lstm(s)
            y[i] = rs[0][len(x[i]) - 1][:]
        y = self.line1(y)
#        print(y)
        y = F.softmax(y, dim=1)
#        print(y)
        return y

    def predict(self, x):
        n = x.shape[0]
        i = 0
        y = []
        for i in range(n):
            p = self.forward(x[i: i + 1]).cpu()
            y.append(p[0])
        y = torch.tensor(np.array([item.detach().numpy() for item in y]))
#        print(y)
        return y


class TreeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(64, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.line = nn.Linear(64, 2)
    
    def forward(self, s, c):
        y = Variable(torch.zeros(0, dtype=torch.float64).cuda(), requires_grad=True).cuda()
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
                    sub = []

                ts = torch.zeros(0, dtype=torch.float64).cuda()
                for x in a:
                    ts = torch.cat((ts, x), 0)
                ts = ts.view([1, len(a), a[0].size()[0]])
                rs, h = self.lstm2(ts)
                f[j] = rs[0][len(a) - 1][:]
            
            rs = self.line(f[0])
            rs = rs.view([1, 2])
            rs = torch.softmax(rs, dim=1)
            y = torch.cat((y, rs), 0)

        return y

    def predict(self, s, c):
        n = len(s)
        i = 0
        y = []
        for i in range(n):
            p = self.forward(s[i: i + 1], c[i: i + 1]).cpu()
            y.append(p[0])
        y = torch.tensor(np.array([item.detach().numpy() for item in y]))
        return y