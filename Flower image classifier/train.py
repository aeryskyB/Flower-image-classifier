import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import argparse

import helper

# ----------------------------------------------------------------
# creating a argument parser for this file
parser = argparse.ArgumentParser(description='train.py')

## assigning other command line arguments
# 1. path to data directory
parser.add_argument('data_dir', action='store')         # default='./flowers'
# 2. path to save the the .pth checkpoint
parser.add_argument('--save_dir', action='store', default='./checkpoint.pth')
# 3. architecture for the model
parser.add_argument('--arch', action='store', default='vgg19', choices=['alexnet', 'densenet121', 'vgg19'])
# 4. learning rate
parser.add_argument('--learning_rate', action='store', default='0.001', type=float, dest='lr')
# 5. number of hidden units of the first hidden layer
parser.add_argument('--hidden_layers', action='store', default='1024', type=int, dest='num_hidden')
# 6. number of epochs
parser.add_argument('--epochs', action='store', default='40', type=int, dest='num_epochs')
# 7. GPU usage
parser.add_argument('--gpu', action='store_true')

# calling the parser
args = parser.parse_args()

# obtaining data from the command line arguments
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
lr = args.lr
num_hidden = args.num_hidden
num_epochs = args.num_epochs
gpu = args.gpu

# ----------------------------------------------------------------

# creating dataloaders
train_dataloader, valid_dataloader, test_dataloader, class_to_idx = helper.load_data(data_dir)
print('Data loaded!')

# calling available architecture mapping
archs = helper.archs

# creating model, loss function, model optimizer, arch (unchanged/changed)
flora, criterion, optimizer, arch = helper.nn_artist(arch, 0.5, num_hidden, lr)
print('Model created!')

# training the model
flora = helper.train_model(flora, criterion, optimizer, train_dataloader, 
                           valid_dataloader, arch, num_epochs, 32, num_hidden, lr, gpu)
print('Model trained!')

# prdicting accuracy
helper.accuracy(flora, test_dataloader, gpu)

# saving various data in the checkpoint
helper.save_checkpoint(arch, flora, optimizer, class_to_idx, num_hidden, save_dir)
print('Checkpoint saved!')

print('___End of train.py___')