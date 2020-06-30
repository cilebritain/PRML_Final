import copy
import random
import numpy as np

def ScanFold(datapath):
    f = open(datapath, "r")
    f.readline()
    data = np.zeros((0, 2))
    for s in f.readlines():
        s_ = s.split(",")
        s_[2].replace('\n','')
        data = np.append(data, np.asarray([[s_[1], s_[2]]]), 0)
    np.random.shuffle(data)
    return data

def Split(data, rate):
    np.random.shuffle(data)
    n = int(data.shape[0] * rate[0])
    train_data = np.asarray([data[i] for i in range(n)])
    n = int(data.shape[0] * rate[1])
    val_data = np.asarray([data[i] for i in range(n)])
    n = int(data.shape[0] * rate[2])
    test_data = np.asarray([data[i] for i in range(n)])
    return train_data, val_data, test_data

def Select(data, n, balance = False, rate = 4.):
    tot = data.shape[0]

    if (balance):
        postive = np.asarray([data[i] for i in range(tot) if float(data[i][1]) > .1])
        negative = np.asarray([data[i] for i in range(tot) if float(data[i][1]) <= .1])
        
        postive_num = int(n * 1 / (1 + rate))
        if postive_num > postive.shape[0]:
            negative_num = n - postive_num
        else:
            negative_num = int(n * rate / (1 + rate))
            postive_num = n - negative_num

        np.random.shuffle(postive)
        np.random.shuffle(negative)
        data = np.asarray([postive[i] for i in range(postive_num)] + [negative[i] for i in range(negative_num)])
    else:
        np.random.shuffle(data)
        data = np.asarray([data[i] for i in range(n)])
    
    return data

def Format(data, one_hot = False, align = False): 
    x = np.zeros(0)
    y = np.zeros(0)
    n = data.shape[0]

    if (one_hot):
        max_len = max([len(data[i][0]) for i in range(n)])
        x = np.zeros((n, max_len ,128))
        for i in range(n):
            l = len(data[i][0])
            for j in range(l):
                x[i][j][ord(data[i][0][j])] = 1.
            for j in range(max_len - l):
                x[i][j + l][0] = 1.
    else:
        max_len = max([len(data[i][0]) for i in range(n)])
        x = np.zeros((0, max_len, 1))
        if align:
            for i in range(data.shape[0]):
                y = [[[ord(data[i][0][j])] for j in range(len(data[i][0]))] + [[0] for j in range(max_len - len(data[i][0]))]]
                x = np.append(x, np.asarray(y), 0)
        else: 
            x = np.asarray([data[i][0] for i in range(n)])

    y = np.asarray([float(data[i][1]) for i in range(n)])
    return x, y
    

