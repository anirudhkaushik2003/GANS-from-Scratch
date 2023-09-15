import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL 
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision

BATCH_SIZE = 64
IMG_SIZE = 32

data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))

])


train = torchvision.datasets.MNIST(root='/ssd_scratch/cvit/anirudhkaushik/datasets/', train=True, download=False, transform=data_transforms)
test = torchvision.datasets.MNIST(root='/ssd_scratch/cvit/anirudhkaushik/datasets/', train=False, download=False, transform=data_transforms)

data_loader = DataLoader(torch.utils.data.ConcatDataset([train, test]), batch_size=BATCH_SIZE, shuffle=True)