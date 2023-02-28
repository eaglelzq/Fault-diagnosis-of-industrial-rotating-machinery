# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年07月02日
"""
from myDataset import Train_Dataset,Test_Dataset
from torch.utils import data as data_
from BaseModels import CustomResnet18
import torch
from torch import nn
from d2l import torch as d2l
import argparse
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
# data = pd.read_csv('total_data.csv')
# X_train, X_test, y_train, y_test = train_test_split(data[['cwt_data','stft_data','gray_data']].values,data['label'].values.tolist(),test_size=0.3)





def train_ch6(net, train_iter,num_epochs, lr,lambd,device):
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (data,y) in enumerate(train_iter):
            timer.start()
            y = torch.tensor(y,dtype=torch.int64)
            data,y= data.to(device), y.to(device)
            y_hat = net(data)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                metric.add(l * data.shape[0], d2l.accuracy(y_hat, y), data.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        print(f'epoch{epoch}:loss {train_l:.3f}, train acc {train_acc:.3f}')
    torch.save(net.state_dict(), './model_saver/Presnet18.params')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

if __name__=='__main__':
    #Step3生成的保存训练集图片的csv文件地址
    Trainset = Train_Dataset(r'D:\Study\competition_lerning\Yidongbei\Data_version\training\total_train.csv')
    print('load data')
    trainloader = data_.DataLoader(Trainset,batch_size=32,shuffle=True)
    Net = CustomResnet18()
    lr = 0.0001
    lambd = 0
    device = 'cuda'
    train_ch6(Net,trainloader,30,lr,lambd,device)
# if __name__ =="__main__":
#     net = CustomResNext()
#     net.load_state_dict(torch.load('D:\Study\competition_lerning\YiDongbei2\model_params\Pinception.params'))
