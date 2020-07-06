import torch
import argparse
from numpy import float64
from handout import *


def ComputeLoss(y, t):
    y = y.cuda()
    t = t.cuda()
    loss_fun = torch.nn.CrossEntropyLoss(reduction='none')
    weight = torch.tensor([1. if x == 1 else .5 for x in t ]).cuda()
    loss = loss_fun(y, t) * weight / t.size()[0]
    loss = loss.sum()
    return loss

def TrainOneStepTree(model, s, c, t, optimizer):
    model = model.double().cuda()
    optimizer.zero_grad()
    t = torch.tensor(t, dtype=torch.long).cuda()
    y = model(s, c).cuda()
    loss = ComputeLoss(y, t)
    loss.backward()
    optimizer.step()

    return loss.item()

def TrainOneStepLstm(model, x, t, optimizer):
    model = model.double().cuda()
    optimizer.zero_grad()
    t = torch.tensor(t, dtype=torch.long).cuda()
    y = model(x).cuda()
    loss = ComputeLoss(y, t)
    loss.backward()
    optimizer.step()

    return loss.item()


def Test(test_data, model, model_kind):
    if model_kind == 'lstm':
        x, t = Format(test_data, one_hot=False, align=False)
        y = model.predict(x)
        t = torch.tensor(t, dtype=torch.long)
        loss = ComputeLoss(y, t)
    else:
        s, c, t = Parentheses(test_data)
        y = model.predict(s, c)
        t = torch.tensor(t, dtype=torch.long)
        loss = ComputeLoss(y, t)
    return loss.cpu()


def Train(train_data, val_data, model, optimizer, epoch = 200, batch_size = 100, model_kind='tree'):
    loss = 0.0
    for step in range(epoch):
        batch_data = Select(train_data, n=batch_size, balance=True, rate=1.5)
        loss = 0.
        if model_kind == 'lstm':
            x, t = Format(batch_data, one_hot=False, align=False)
            loss = TrainOneStepLstm(model, x, t, optimizer)
        else:
            s, c, t = Parentheses(batch_data)
            loss = TrainOneStepTree(model, s, c, t, optimizer)
        if step % 20 == 0:
            loss = Test(val_data, model, model_kind)
            print('step', step, ': loss = ', loss)


def CrossValidate(datapath, model_kind):
    for i in range(10):
        if model_kind == 'tree':
            model = TreeModel()
        else:
            model = LSTMModel()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)

        train_data = ScanFold(datapath + '/fold_' + str(i) + '/train.csv')
        val_data = ScanFold(datapath + '/fold_' + str(i) + '/dev.csv')
        test_data = ScanFold(datapath + '/fold_' + str(i) + '/test.csv')

        Train(train_data, val_data, model, optimizer, model_kind=model_kind)
        loss = Test(test_data, model, model_kind)
        print('test' + str(i) + ': loss = ', loss.item())

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../data/train_cv')
#    parser.add_argument('--save_dir', type=str, default='./checkpoint/checkpoint')  
    parser.add_argument('--model_kind', type=str, choices=['lstm', 'tree'], default='lstm')
    arg = parser.parse_args()

    CrossValidate(datapath=arg.data_path, model_kind=arg.model_kind)