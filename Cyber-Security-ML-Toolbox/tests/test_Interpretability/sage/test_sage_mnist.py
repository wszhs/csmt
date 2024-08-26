import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import torch
import torchvision.datasets as dsets

# Load train set
train = dsets.MNIST('../', train=True, download=True)
imgs = train.data.reshape(-1, 784) / 255.0
labels = train.targets

# Shuffle and split into train and val
inds = torch.randperm(len(train))
imgs = imgs[inds]
labels = labels[inds]
val, Y_val = imgs[:6000], labels[:6000]
train, Y_train = imgs[6000:], labels[6000:]

# Load test set
test = dsets.MNIST('../', train=False, download=True)
test, Y_test = test.data.reshape(-1, 784) / 255.0, test.targets

# Move test data to numpy
test_np = test.cpu().data.numpy()
Y_test_np = Y_test.cpu().data.numpy()

import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader

# Create model
device = torch.device('cpu', 0)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ELU(),
    nn.Linear(256, 256),
    nn.ELU(),
    nn.Linear(256, 10)).to(device)

# Training parameters
lr = 1e-3
mbsize = 64
max_nepochs = 250
loss_fn = nn.CrossEntropyLoss()
lookback = 5
verbose = False

# Move to GPU
train = train.to(device)
val = val.to(device)
test = test.to(device)
Y_train = Y_train.to(device)
Y_val = Y_val.to(device)
Y_test = Y_test.to(device)

# Data loader
train_set = TensorDataset(train, Y_train)
train_loader = DataLoader(train_set, batch_size=mbsize, shuffle=True)

# Setup
optimizer = optim.Adam(model.parameters(), lr=lr)
min_criterion = np.inf
min_epoch = 0

# Train
for epoch in range(max_nepochs):
    for x, y in train_loader:
        # Move to device.
        x = x.to(device=device)
        y = y.to(device=device)

        # Take gradient step.
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        model.zero_grad()

    # Check progress.
    with torch.no_grad():
        # Calculate validation loss.
        val_loss = loss_fn(model(val), Y_val).item()
        if verbose:
            print('{}Epoch = {}{}'.format('-' * 10, epoch + 1, '-' * 10))
            print('Val loss = {:.4f}'.format(val_loss))

        # Check convergence criterion.
        if val_loss < min_criterion:
            min_criterion = val_loss
            min_epoch = epoch
            best_model = deepcopy(model)
        elif (epoch - min_epoch) == lookback:
            if verbose:
                print('Stopping early')
            break

# Keep best model
model = best_model