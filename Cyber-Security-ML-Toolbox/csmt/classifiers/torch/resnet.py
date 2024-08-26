# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/101_models.ResNet.ipynb (unless otherwise specified).

__all__ = ['ResBlock', 'ResNet']

# Cell
import torch.nn as nn
from .layers import *
from fastai.basics import *
from csmt.classifiers.abstract_model import AbstractModel
import torch
import torch.nn as nn
from csmt.classifiers.abstract_model import AbstractModel
from csmt.estimators.classification.pytorch import PyTorchClassifier

# Cell
class ResBlock(Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        self.convblock1 = ConvBlock(ni, nf, kss[0])
        self.convblock2 = ConvBlock(nf, nf, kss[1])
        self.convblock3 = ConvBlock(nf, nf, kss[2], act=None)

        # expand channels for the sum if necessary
        self.shortcut = BN1d(ni) if ni == nf else ConvBlock(ni, nf, 1, act=None)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x

class ResNet(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResNet,self).__init__()
        nf = 64
        kss=[7, 5, 3]
        self.resblock1 = ResBlock(c_in, nf, kss=kss)
        self.resblock2 = ResBlock(nf, nf * 2, kss=kss)
        self.resblock3 = ResBlock(nf * 2, nf * 2, kss=kss)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.fc = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.squeeze(self.gap(x))
        return self.fc(x)

class ResNetTorch(AbstractModel):

    def __init__(self, input_size,learning_rate=1e-3,
                weight_decay=0,output_size=None):
        model=ResNet(c_in=input_size, c_out=output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size,clip_values=(0,1))
