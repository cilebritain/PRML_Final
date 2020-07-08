import math
import torch
import argparse
from numpy import float64
from handout import *
from sklearn import metrics
from torch.autograd import Variable

def TrainOneStepTree(model, s, c, t, optimizer):
    model = model.double().cuda()
    optimizer.zero_grad()
    t = torch.tensor(t, dtype=torch.long).cuda()
    y = model(s, c).cuda()
    fl = FocalLoss(gamma=2).cuda()
    loss = fl(y, t)
    loss.backward()
    optimizer.step()

    return loss.item()

def TrainOneStepLstm(model, x, t, optimizer):
    model = model.double().cuda()
    optimizer.zero_grad()
    t = torch.tensor(t, dtype=torch.long).cuda()
    y = model(x).cuda()
    fl = FocalLoss(gamma=2).cuda()
    loss = fl(y, t)
    loss.backward()
    optimizer.step()

    return loss.item()

def Evaluate(t, y):
    truth = []
    prediction = []
    truth = [int(x) for x in t]
    prediction = [float(y[i][1]) for i in range(y.size()[0])]
    roc_auc = metrics.roc_auc_score(truth, prediction)
    p, r, thr = metrics.precision_recall_curve(truth, prediction)
    prc_auc = metrics.auc(r, p)
    return roc_auc, prc_auc

def Test(test_data, model, model_kind):
    if model_kind == 'lstm':
        x, t = Format(test_data, one_hot=False, align=False)
        y = model.predict(x)
        t = torch.tensor(t, dtype=torch.long)
#        loss = ComputeLoss(y, t)
        roc_auc, prc_auc = Evaluate (t, y)
    else:
        s, c, t = Parentheses(test_data)
        y = model.predict(s, c)
        t = torch.tensor(t, dtype=torch.long)
#        loss = ComputeLoss(y, t)
        roc_auc, prc_auc = Evaluate (t, y)
        
#    return loss.cpu()
    return roc_auc, prc_auc

def Train(train_data, val_data, model, optimizer, epoch = 200, batch_size = 100, model_kind='tree'):
    loss = 0.0
    for step in range(epoch):
        batch_data = Select(train_data, n=batch_size, balance=True, rate=3.)
        loss = 0.
        if model_kind == 'lstm':
            x, t = Format(batch_data, one_hot=False, align=False)
            loss = TrainOneStepLstm(model, x, t, optimizer)
        else:
            s, c, t = Parentheses(batch_data)
            loss = TrainOneStepTree(model, s, c, t, optimizer)
        if step % 30 == 0:
            roc_auc, prc_auc = Test(val_data, model, model_kind)
            print('step', step, ': roc_auc =', roc_auc, ', prc_auc =', prc_auc)


def CrossValidate(datapath, model_kind, fold_num=0):
    i = fold_num
    if model_kind == 'tree':
        model = TreeModel()
    else:
        model = LSTMModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)

    train_data = ScanFold(datapath + '/fold_' + str(i) + '/train.csv')
    val_data = ScanFold(datapath + '/fold_' + str(i) + '/dev.csv')
    test_data = ScanFold(datapath + '/fold_' + str(i) + '/test.csv')

    Train(train_data, val_data, model, optimizer, model_kind=model_kind)
    roc_auc, prc_auc = Test(test_data, model, model_kind)

    if model_kind == 'lstm':
        x, t = Format(test_data, one_hot=False, align=False)
        y = model.predict(x)
#        print(y)
    else:
        s, c, t = Parentheses(test_data)
        y = model.predict(s, c)
        print(y)

    print('test' + str(i) + ': roc_auc = ' + str(roc_auc) + ', prc_auc = ' + str(prc_auc))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../data/train_cv')
#    parser.add_argument('--save_dir', type=str, default='./checkpoint/checkpoint')  
    parser.add_argument('--model_kind', type=str, choices=['lstm', 'tree'], default='lstm')
    parser.add_argument('--fold_num', type=int, default=0)
    arg = parser.parse_args()

    CrossValidate(datapath=arg.data_path, model_kind=arg.model_kind, fold_num=arg.fold_num)