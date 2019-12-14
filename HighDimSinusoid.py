from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import copy
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv



parser = argparse.ArgumentParser(description='VAE MNIST Example')

parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=128, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

device = torch.device("cuda" if args.cuda else "cpu")


def loss_function(recon_x, x):
    return F.nll_loss(recon_x, x)


class MINI(nn.Module):
    def __init__(self):
        super(MINI, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        s = torch.cat((torch.zeros(x.size()).to(device), x), dim=1)
        return F.log_softmax(s, dim=1)


class LinCL(nn.Module):
    def __init__(self):
        super(LinCL, self).__init__()
        self.fc1 = nn.Linear(100, 1)
        self.b = 0

    def forward(self, x):
        a = self.fc1(x)
        s = torch.cat((torch.zeros(a.size()).to(device), a), dim=1)
        return F.log_softmax(s, dim=1)


MINI_train = MINI().to(device)
LinCL_train = LinCL().to(device)

optimizer = optim.SGD(MINI_train.parameters(), lr=1e-3)
linopt = optim.SGD(LinCL_train.parameters(), lr=1e-3)

w=torch.randn(100).to(device)
w=w/w.norm()

w_=torch.randn(100).to(device)
w_ortho = w_- w* (w.matmul(w_)/w.norm().pow(2))

scheduler=StepLR(optimizer, step_size=1, gamma=0.94)
lin_scheduler=StepLR(linopt, step_size=1, gamma=0.94)

filewriter= open('MutualInfo.csv', "w")
writer = csv.writer(filewriter)
writer.writerow(['IFYL', 'IFY', 'correct', 'neural_correct'])

for epoch in range(1, 200):
    for batch_idx in range(0, 2000):
        data = torch.randn(args.batch_size, 100).to(device)
        y = (torch.clamp(torch.sign(data.matmul(w)
                                    +torch.sin(data.matmul(w_ortho))), min=0)).type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        linopt.zero_grad()
        recon_y= MINI_train(data)
        lin_y = LinCL_train(data)
        loss = loss_function(recon_y, y)
        lin_loss = loss_function(lin_y, y)
        loss.backward()
        lin_loss.backward()
        optimizer.step()
        linopt.step()
        if batch_idx%100==0:
            print('loss : {}, linloss : {}'.format(loss, lin_loss))
    print('----------------------epoch : {}----------------------'.format(epoch))

    result = torch.zeros(2, 2, 2)
    IFYL=0
    IFY =0
    correct = 0
    neural_correct = 0
    for batch_idx in range(0,600):
        data = torch.randn(args.batch_size, 100).to(device)
        y = (torch.clamp(torch.sign(data.matmul(w)+torch.sin(data.matmul(w_ortho))),min=0)).type(torch.LongTensor).to(device)
            #f,y,g order
        f_y = torch.exp(MINI_train(data))
        g_y = torch.exp(LinCL_train(data))
	
        #ff = f_y.argmax(dim=1,keepdim=True)
        #gg = g_y.argmax(dim=1,keepdim=True)
        #fyg = torch.cat((ff, y.view(y.size()[0], 1), gg), dim=1)

        for i in range(0, y.size()[0]):
        #    result[fyg[i][0], fyg[i][1], fyg[i][2]] += 1
		result[0,y[i],0] += (f_y[i][0])*(g_y[i][0])
		result[1,y[i],0] += f_y[i][1]*(g_y[i][0])
		result[0,y[i],1] += (f_y[i][0])*g_y[i][1]
		result[1,y[i],1] += f_y[i][1]*g_y[i][1]

    total = result.sum()

    for fff in range(0, 2):
        for yyy in range(0, 2):
            pfy = result[fff, yyy, :].sum() / total
            pf = result[fff, :, :].sum() / total
            py = result[:, yyy, :].sum() / total
            IFY += pfy*torch.log(pfy/pf/py)
            for ggg in range(0, 2):
                pfyg=result[fff, yyy, ggg]/total
                pfg = result[fff, :, ggg].sum()/total
                pyg = result[:, yyy, ggg].sum()/total
                pg = result[:, :, ggg].sum()/total
                IFYL += pfyg*torch.log(pfyg*pg/pfg/pyg)
                if yyy==ggg:
                    correct+=pfyg
                if fff==yyy:
                    neural_correct+=pfyg

    print('IFYL : {}, IFY : {}, correct : {}, neural : {}'.format(IFYL, IFY,correct, neural_correct))

    writer.writerow([IFYL.item(), IFY.item(), correct.item(), neural_correct.item()])
    scheduler.step()
    lin_scheduler.step()
