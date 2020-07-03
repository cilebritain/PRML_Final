import torch
import argparse
from numpy import float64
from handout import *

def TrainOneStepTree(model, s, c, t, optimizer):
    t = torch.tensor(t, dtype=torch.long).cuda()
#    x = torch.tensor(x).cuda()
    model = model.double().cuda()
    
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    y = model(s, c).cuda()
#    print(y.cpu())
#    print(t)
    loss = loss_fun(y, t)    
    loss.backward()
    optimizer.step()

    return loss.item()

def TrainOneStepLstm(model, x, t, optimizer):
    t = torch.tensor(t, dtype=torch.long).cuda()
    model = model.double().cuda()
    
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    y = model(x).cuda()
    loss = loss_fun(y, t)    
    loss.backward()
    optimizer.step()

    return loss.item()

def Train(train_data, model, optimizer, epoch = 500, batch_size = 150, model_kind='tree'):
    loss = 0.0
    for step in range(epoch):
        batch_data = Select(train_data, n=batch_size, balance=True, rate=1.5)
        loss = 0.
        if model_kind == 'lstm':
            x, t = Format(batch_data, one_hot=False, align=False)
#            print(x, t)
#            return
            loss = TrainOneStepLstm(model, x, t, optimizer)
        else:
            s, c, t = Parentheses(batch_data)
            loss = TrainOneStepTree(model, s, c, t, optimizer)
        print('step', step, ': loss', loss)

def Work(datapath, save_dir, model_kind):
    all_data = data.ScanFold(datapath)

    if model_kind == 'lstm':
        lstmmodel = model.LSTMModel()
        optimizer = torch.optim.Adam(params=lstmmodel.parameters(), lr=0.005)
        Train(all_data, model=lstmmodel, optimizer=optimizer, model_kind='lstm')
        state = {'net':lstmmodel.state_dict(), 'model_kind':'lstm'}
        torch.save(state, save_dir)
    else:
        treemodel = model.TreeModel()
        optimizer = torch.optim.Adam(params=treemodel.parameters(), lr=0.005)
        Train(all_data, model=treemodel, optimizer=optimizer, model_kind='tree')
        state = {'net':treemodel.state_dict(), 'model_kind':'tree'}
        torch.save(state, save_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train.csv')
    parser.add_argument('--save_dir', type=str, default='./checkpoint/checkpoint')  
    parser.add_argument('--model_kind', type=str, choices=['lstm', 'tree'], default='tree')
    arg = parser.parse_args()

    Work(datapath=arg.data_path, save_dir=arg.save_dir, model_kind=arg.model_kind)