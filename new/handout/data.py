import copy
import random
import numpy as np

def ScanFold(datapath):
    f = open(datapath, "r")
    f.readline()
    data = np.zeros((0, 2))
    for s in f.readlines():
        s_ = s.split(",")
        s_[2] = s_[2].replace('\n','')
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
    '''
    Select subdata from data
    Select postive samples : negative samples (0.1) euqals to rate if balace
    '''
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
    '''
    Convert string into ord list or one hot list
    expend the list into the same length(max) if align
    '''
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

    y = np.asarray([int(data[i][1]) for i in range(n)])
    return x, y
    
def Parentheses(data):
    '''
    Bulid a tree structure from the string
    c[i][j][k][0..1]: in the sample i and the node j, the node(0) poings to and the postion the substring in the string j(1)
    s[i][j]: the string in the sample i and the node j
    y[i]: the ground truth of the sample i
    '''
    y = np.zeros(0)
    n = data.shape[0] 
    s = []
    c = []

    for i in range(n):
        smiles = '(' + data[i][0]+ ')'
        top = -1
        N = 0
        s.append([])
        c.append([])
        stack = [0 for i in range(len(smiles))]

        for j in range(len(smiles)):
            if smiles[j] == '(':
                p = stack[top] if top >= 0 else 0
                top = top + 1
                stack[top] = N
                N = N + 1

                c[i].append([])
                s[i].append('')
                if (top > 0):
                    c[i][p].append([N - 1, len(s[i][p])])
            else:
                if smiles[j] == ')':
                    stack[top] = 0
                    top = top - 1
                else:
                    s[i][stack[top]] += smiles[j]
                
    y = np.asarray([float(data[i][1]) for i in range(n)])
    return s, c, y