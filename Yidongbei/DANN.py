# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年07月25日
"""
import torch.nn as nn

from BaseModels import Resnet18

from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANNET(nn.Module):
    def __init__(self,num_class=8):
        super(DANNET, self).__init__()
        self.featureNet = Resnet18()
        self.bottleneck = nn.Sequential(nn.Linear(512,256),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())
        self.source_fc = nn.Sequential(nn.Linear(256,num_class),
                                       nn.LogSoftmax())

        # self.softmax =  nn.LogSoftmax()
        self.classes = num_class
        #全局域判别器
        self.domain_classfier = nn.Sequential()
        self.domain_classfier.add_module('fc1',nn.Linear(256,1024))
        self.domain_classfier.add_module('bn1', nn.BatchNorm1d(1024))
        self.domain_classfier.add_module('relu1',nn.ReLU())
        self.domain_classfier.add_module('drop1',nn.Dropout())
        self.domain_classfier.add_module('fc2',nn.Linear(1024,1024))
        self.domain_classfier.add_module('bn2', nn.BatchNorm1d(1024))
        self.domain_classfier.add_module('relu2',nn.ReLU())
        self.domain_classfier.add_module('drop2',nn.Dropout())
        self.domain_classfier.add_module('fc3',nn.Linear(1024,2))
        self.domain_classfier.add_module('d_softmax', nn.LogSoftmax())

    def forward(self,x,alpha=0.0):
        x_share = self.featureNet(x)
        x_share = self.bottleneck(x_share)
        x_task = self.source_fc(x_share)
        x_reverse = ReverseLayerF.apply(x_share,alpha)
        x_domain = self.domain_classfier(x_reverse)
        return x_task,x_domain
    def predict(self,x):
        x = self.featureNet(x)
        x = self.bottleneck(x)
        x = self.source_fc(x)
        return x



