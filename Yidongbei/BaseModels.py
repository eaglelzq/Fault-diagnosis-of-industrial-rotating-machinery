# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年07月25日
"""
# from pip._internal import main
# main(['install', 'timm'])
import torch.nn as nn
import timm
import torch

class CustomResnet18(nn.Module):
    def __init__(self,model_name='resnet18',pretrained=True):
        super(CustomResnet18, self).__init__()
        self.model = timm.create_model(model_name,pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.conv1 = nn.Conv2d(6,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(n_features,8,bias=True)
    def forward(self,x):
        x = self.model(x)
        return x
class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        resnet = CustomResnet18()
        resnet.load_state_dict(torch.load('./model_saver/Presnet18.params'))
        resnet = resnet.model
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.act1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.global_pool
        self._feature_dim = resnet.fc.in_features
        del resnet
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    def output_num(self):
        return self._feature_dim