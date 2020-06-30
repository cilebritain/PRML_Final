import torch
import argparse
from numpy import float64
from handout import *

def TrainOneStep(model, x, t, optimizer):
    t = torch.tensor(t).cuda()
    x = torch.tensor(x).cuda()
    model = model.double().cuda()
    
    loss_fun = torch.nn.MSELoss(reduction='none')
    weight = torch.tensor([1 if t[i] > .1 else .25 for i in range(x.size()[0])]).cuda()
    optimizer.zero_grad()
    y = model(x).cuda()
    loss = loss_fun(y, t)
    loss = loss * weight
    loss = torch.sum(loss) / loss.size()[0]
    loss.backward()
    optimizer.step()

    return loss.item()

def Train(train_data, model, optimizer, epoch = 100, batch_size = 100):
    loss = 0.0
    for step in range(epoch):
        batch_data = data.Select(train_data, n=batch_size, balance=True, rate=5.)
        x, t = data.Format(batch_data, one_hot=False, align=True)
        loss = TrainOneStep(model, x, t, optimizer)
        print('step', step, ': loss', loss)

def Work(datapath, save_dir):
    all_data = data.ScanFold(datapath)

    lstmmodel = model.LSTMModel()
    optimizer = torch.optim.Adam(params=lstmmodel.parameters(), lr=0.001)
    Train(all_data, model=lstmmodel, optimizer=optimizer)

    state = {'net':lstmmodel.state_dict()}
    torch.save(state, save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train.csv')
    parser.add_argument('--save_dir', type=str, default='./checkpoint/checkpoint')  
    arg = parser.parse_args()

    Work(datapath=arg.data_path, save_dir=arg.save_dir)