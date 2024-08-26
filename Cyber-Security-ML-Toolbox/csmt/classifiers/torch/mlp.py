from csmt.classifiers.abstract_model import AbstractModel
import torch
import torch.nn as nn
import numpy as np
from csmt.estimators.classification.pytorch import PyTorchClassifier
import torch.nn.functional as F
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    def __init__(self,input_size,output_size):
        super(MLP,self).__init__()
        self.fc_1=nn.Linear(in_features=input_size,out_features=32)
        self.fc_2=nn.Linear(in_features=32,out_features=10)
        self.fc_3=nn.Linear(in_features=10,out_features=output_size)
    def forward(self,x):
        x=self.fc_1(x)
        x=torch.relu(x)
        x=self.fc_2(x)
        x=torch.relu(x)
        x = self.fc_3(x)
        x=torch.sigmoid(x)
        return x
        
class AlertNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(AlertNet,self).__init__()
        self.fc_1=nn.Linear(in_features=input_size,out_features=1024)
        self.bn_1 = nn.BatchNorm1d(1024)
        self.fc_2=nn.Linear(in_features=1024,out_features=768)
        self.bn_2 = nn.BatchNorm1d(768)
        self.fc_3=nn.Linear(in_features=768,out_features=512)
        self.bn_3 = nn.BatchNorm1d(512)
        self.fc_4=nn.Linear(in_features=512,out_features=256)
        self.bn_4 = nn.BatchNorm1d(256)
        self.fc_5=nn.Linear(in_features=256,out_features=128)
        self.bn_5 = nn.BatchNorm1d(128)
        self.fc_6=nn.Linear(in_features=128,out_features=output_size)
        
    def forward(self,x):
        x=self.fc_1(x)
        x = self.bn_1(x)
        x=F.dropout(x,p=0.01)
        x=self.fc_2(x)
        x = self.bn_2(x)
        x=F.dropout(x,p=0.01)
        x = self.fc_3(x)
        x = self.bn_3(x)
        x=F.dropout(x,p=0.01)
        x = self.fc_4(x)
        x = self.bn_4(x)
        x=F.dropout(x,p=0.01)
        x = self.fc_5(x)
        x = self.bn_5(x)
        x=F.dropout(x,p=0.01)
        x = self.fc_6(x)
        x=torch.sigmoid(x)
        return x

class DeepNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(DeepNet,self).__init__()
        self.fc_1=nn.Linear(in_features=input_size,out_features=256)
        self.fc_2=nn.Linear(in_features=256,out_features=256)
        self.fc_3=nn.Linear(in_features=256,out_features=256)
        self.fc_4=nn.Linear(in_features=256,out_features=256)
        self.fc_5=nn.Linear(in_features=256,out_features=output_size)
    def forward(self,x):
        x=self.fc_1(x)
        x=F.dropout(x,p=0.01)
        x=self.fc_2(x)
        x=F.dropout(x,p=0.01)
        x = self.fc_3(x)
        x=F.dropout(x,p=0.01)
        x = self.fc_4(x)
        x=F.dropout(x,p=0.01)
        x = self.fc_5(x)
        x=torch.sigmoid(x)
        return x
    
class IdsNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(IdsNet,self).__init__()
        self.fc_1=nn.Linear(in_features=input_size,out_features=123)
        self.fc_2=nn.Linear(in_features=123,out_features=64)
        self.fc_3=nn.Linear(in_features=64,out_features=5)
        self.fc_4=nn.Linear(in_features=5,out_features=output_size)
    def forward(self,x):
        x=self.fc_1(x)
        x=torch.relu(x)
        x=self.fc_2(x)
        x=torch.relu(x)
        x =self.fc_3(x)
        x=torch.relu(x)
        x =self.fc_4(x)
        x=torch.sigmoid(x)
        return x
    
class MLPTorch(AbstractModel):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size,learning_rate=0.01,
                weight_decay=0,output_size=None):
        model=MLP(input_size=input_size,output_size=output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size,clip_values=(0,1))
        
        
class IdsNetTorch(AbstractModel):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size,learning_rate=0.01,
                weight_decay=0,output_size=None):
        model=IdsNet(input_size=input_size,output_size=output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size,clip_values=(0,1))

class DeepNetTorch(AbstractModel):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size,learning_rate=0.01,
                weight_decay=0,output_size=None):
        model=DeepNet(input_size=input_size,output_size=output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size,clip_values=(0,1))

class AlertNetTorch(AbstractModel):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size,learning_rate=0.01,
                weight_decay=0,output_size=None):
        model=AlertNet(input_size=input_size,output_size=output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size,clip_values=(0,1))



