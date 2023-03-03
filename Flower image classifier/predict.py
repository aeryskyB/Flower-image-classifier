import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import argparse

import helper

import json

# ----------------------------------------------------------------
# creating a argument parser for this file
parser = argparse.ArgumentParser(description='predict.py')

## assigning other command line arguments
# 1. path to image file
parser.add_argument('input', action='store')
# 2. path to .pth checkpoint
parser.add_argument('checkpoint', action='store')
# 3. top k number of classes
parser.add_argument('--top_k', action='store', default=5, type=int)
# 4. .json for real category names
parser.add_argument('--category_names', action='store', default='cat_to_name.json')
# 5. using GPU
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

image = args.input
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

# ----------------------------------------------------------------
# getting the map of index to categories from the .json file
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
print('Categories loaded.')

# ----------------------------------------------------------------
# model loading
flora,_ = helper.load_model_optimizer(checkpoint)
print('Model loaded')

# getting probabilities and classes
ps, cs = helper.predict(image, flora, top_k, gpu)
# moving to cpu
device = torch.device('cpu')
ps, cs = ps.to(device), cs.to(device)
ps, cs = np.array(ps[0]), np.array(cs[0])
print('Probabilities obtained.')

# finding corresponding category to every class
idx_to_class = {i : c for c, i in flora.class_to_idx.items()}
classes = [idx_to_class[c] for c in cs]
names = [cat_to_name[c] for c in classes]

# printing to the console
for i in range(len(names)):
    print(f'Probability being *{names[i]}*: {ps[i]:0.3f}')

print('___End of predict.py___')