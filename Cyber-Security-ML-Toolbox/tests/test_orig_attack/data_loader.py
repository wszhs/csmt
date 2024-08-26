'''
Author: your name
Date: 2021-03-28 16:51:49
LastEditTime: 2021-06-16 10:24:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Mnist_Attack-main/data_loader.py
'''
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

train_batch_size = 64
test_batch_size = 128

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST('tests/test_orig_attack/mnist_data', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST('./mnist_data', train = False, transform=transform)
test_dataset = datasets.MNIST('tests/test_orig_attack/mnist_data', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

print("data loading is done!")
