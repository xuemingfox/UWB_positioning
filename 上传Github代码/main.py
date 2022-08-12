# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 18:30:47 2021

@author: VCG-group
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
from net import Net
from tqdm import tqdm


num_inputs = 16
batch_size = 32
hidden_size = 200
label_size = 3
num_epochs = 100
learning_rate = 3e-3
eta_min = 1e-8


def load(data_dir):

    file_train = open(data_dir, "r")
    train_data = file_train.read().split("\n")[:-1]
    file_train.close()

    print(len(train_data))

    fea = []
    lab = []
    for i in train_data:
        k = i.split(":")
        k = [int(j) for j in k]
        fea.append(k[:-3])
        lab.append(k[-3:])

    fea = torch.tensor(fea, dtype=torch.float)
    lab = torch.tensor(lab, dtype=torch.float)
    features = fea
    labels = lab
    dataset = Data.TensorDataset(features, labels)
    # 随机读取小批量
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    return data_iter

def showLR(opt):
    lr = []
    for param_group in opt.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)






train_iter = load("data/normal_p_train.txt")
test_iter = load("data/normal_p_test.txt")
# features = torch.zeros(100,4, dtype=torch.float)
# labels = torch.zeros(100,3, dtype=torch.float)



net = Net(num_inputs, hidden_size, label_size, t="ResNet").cuda()




loss = nn.MSELoss()


optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_epochs), eta_min=eta_min)


def train(model, data_iter):

    for epoch in range(1, num_epochs + 1):
        model.train()
        loss_sum = 0
        LR = showLR(optimizer)
        print("current lr=", LR)
        for X, y in tqdm(data_iter):
            X = X.cuda()
            y = y.cuda()
            output = model(X)
            # print(111)
            l = loss(output, y.view(-1, 3))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # print('epoch %d, loss: %f' % (epoch, l.item()/batch_size))
            loss_sum += l.item()
        print("\ntesting...")
        test(net, test_iter)

        print("\nepoch:", epoch, " total loss:", loss_sum / len(data_iter))
        scheduler.step()

def test(model, data_iter):
    loss_sum = 0
    model.eval()

    for X, y in tqdm(data_iter):
        X = X.cuda()
        y = y.cuda()
        output = model(X)
        l = loss(output, y.view(-1, 3)).item()
        loss_sum += l

    print('\ntest loss: %f' % (loss_sum / len(data_iter)))


train(net, train_iter)


