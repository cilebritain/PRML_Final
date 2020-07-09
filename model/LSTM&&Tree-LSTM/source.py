import math
import torch
import argparse
import torch.nn 
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
        if step % 20 == 0:
            roc_auc, prc_auc = Test(val_data, model, model_kind)
            print('step', step, ': roc_auc =', roc_auc, ', prc_auc =', prc_auc)


def Predict(datapath, model, kind, preds_dir):
    test_data = ScanFold(datapath + '/test.csv')
    if kind == 'lstm':
        x, t = Format(test_data, one_hot=False, align=False)
        y = model.predict(x)
    else:
        s, c, t = Parentheses(test_data)
        y = model.predict(s, c)

    f = open(preds_dir, "w")
    f.write('smiles,activity\n')
    for i in range(y.size()[0]):
        f.write(test_data[i][0] + ',' + str(float(y[i][1].data)) + '\n')
    f.close()

def CrossValidate(datapath, model_kind, preds_dir):
    if model_kind == 'tree':
        model = TreeModel()
    else:
        model = LSTMModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)

    train_data = ScanFold(datapath + '/train.csv')
    val_data = ScanFold(datapath + '/dev.csv')

    Train(train_data, val_data, model, optimizer, model_kind=model_kind)

    Predict(datapath, model, model_kind, preds_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/train_cv/fold_1')
    parser.add_argument('--model', type=str, choices=['lstm', 'tree'], default='tree')
    parser.add_argument('--preds_dir', type=str, default='../../data/train_cv/fold_1/preds.csv')
    arg = parser.parse_args()

    CrossValidate(datapath=arg.data_dir, model_kind=arg.model, preds_dir=arg.preds_dir)