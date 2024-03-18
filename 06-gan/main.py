import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import train_loop
from utils.settings import same_seeds, get_device

if __name__ == '__main__':
    same_seeds(2024)
    device = get_device()
    train_loop(device)
