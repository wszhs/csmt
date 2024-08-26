import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import torch
import torchvision.datasets as dsets
# Load train set
train = dsets.MNIST('../', train=True, download=True)
X_train = train.data.reshape(-1, 784).float() / 255.0
Y_train = train.targets.long()
num_features = X_train.shape[1]

# Create validation set
ordering = torch.randperm(len(X_train))
X_train = X_train[ordering]
Y_train = Y_train[ordering]
X_train, X_val = X_train[6000:], X_train[:6000]
Y_train, Y_val = Y_train[6000:], Y_train[:6000]

# Load test set
test = dsets.MNIST('../', train=False, download=True)
X_test = test.data.reshape(-1, 784).float() / 255.0
Y_test = test.targets.long()

print(X_test.shape)

from csmt.Interpretability import rexplain
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from rexplain.torch import MaskLayer2d
from copy import deepcopy

# Prepare device
device = torch.device('cpu')

# Prepare validation data
X_val_missing = X_val.repeat(100, 1)
Y_val_missing = Y_val.repeat(100)

# Random subsets
S_val = torch.ones(X_val_missing.shape)
num_included = np.random.choice(num_features + 1, size=len(S_val))
for i in range(len(S_val)):
    S_val[i, num_included[i]:] = 0
    S_val[i] = S_val[i, torch.randperm(num_features)]

# Create dataset iterator
val_set = TensorDataset(X_val_missing, Y_val_missing, S_val)
val_loader = DataLoader(val_set, batch_size=2056)

def validate(model):
    '''Measure performance on validation set.'''
    with torch.no_grad():
        # Setup
        mean_loss = 0
        N = 0

        # Iterate over validation set
        for x, y, S in val_loader:
            x = x.to(device)
            y = y.to(device)
            S = S.to(device)
            # print(x.shape)
            # print(x.view(-1, 1, 28, 28).shape)
            pred = model((x.view(-1, 1, 28, 28),
                          S.view(-1, 1, 28, 28)))
            loss = loss_fn(pred, y)
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss

# Set up model
model = nn.Sequential(
    MaskLayer2d(),
    nn.Conv2d(2, 16, 3, 1),
    nn.ELU(inplace=True),
    nn.Conv2d(16, 32, 3, 1),
    nn.ELU(inplace=True),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, 3, 1),
    nn.ELU(inplace=True),
    nn.Conv2d(64, 128, 3, 1),
    nn.ELU(inplace=True),
    nn.MaxPool2d(2, 2),
    nn.Flatten(start_dim=1),
    nn.Linear(2048, 256),
    nn.ELU(inplace=True),
    nn.Linear(256, 10)).to(device)


# Training parameters
lr = 1e-3
nepochs = 10
early_stop_epochs = 10

# Loss function
loss_fn = nn.CrossEntropyLoss()
loss_list = []

mbsize=128
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Set up data loaders
train_set = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_set, batch_size=mbsize,
                            shuffle=True, drop_last=True)

# print(train_loader.dataset.data.shape)

# For saving best model
min_epoch = 0

# Begin training
for epoch in range(nepochs):
    for i, (x, y) in enumerate(train_loader):
        # Prepare data
        x = x.to(device)
        y = y.to(device)

        # Generate subset
        S = torch.ones(mbsize, num_features, dtype=torch.float32, device=device)
        num_included = np.random.choice(num_features + 1, size=mbsize)
        for j in range(mbsize):
            S[j, num_included[j]:] = 0
            S[j] = S[j, torch.randperm(num_features)]

        # Make predictions
        pred = model((x.view(-1, 1, 28, 28), S.view(-1, 1, 28, 28)))
        loss = loss_fn(pred, y)

        # Optimizer step
        loss.backward()
        optimizer.step()
        model.zero_grad()

    # End of epoch progress message
    val_loss = validate(model).item()
    loss_list.append(val_loss)
    print('----- Epoch = {} -----'.format(epoch + 1))
    print('Val loss = {:.4f}'.format(val_loss))
    print('')

