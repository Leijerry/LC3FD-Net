# from torchvision.datasets import VOCSegmentation
from torch.utils import data
import torch
import numpy as np
import matplotlib.pyplot as plt
from modelnet40 import ModelNetDataLoader

def get_loader(opts):
    train_set = ModelNetDataLoader(opts.root_dataset, opts, split='train')
    val_set = ModelNetDataLoader(opts.root_dataset, opts, split='test')
    train_loader = data.DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=opts.batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader
