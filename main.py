import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL 
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision

import os
from generator import Generator
from discriminator import Discriminator
from utils import weights_init

import torchvision.utils as vutils

BATCH_SIZE = 64
IMG_SIZE = 32

data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))

])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = torchvision.datasets.MNIST(root='/ssd_scratch/cvit/anirudhkaushik/datasets/', train=True, download=False, transform=data_transforms)
test = torchvision.datasets.MNIST(root='/ssd_scratch/cvit/anirudhkaushik/datasets/', train=False, download=False, transform=data_transforms)

data_loader = DataLoader(torch.utils.data.ConcatDataset([train, test]), batch_size=BATCH_SIZE, shuffle=True)


criterion = nn.BCELoss()

modelG = Generator(IMG_SIZE)
modelD = Discriminator()

modelG = modelG.to(device)
modelD = modelD.to(device)

modelG.apply(weights_init)
modelD.apply(weights_init)

fixed_noise = torch.randn(BATCH_SIZE, 100, 1, 1, device='cuda')
real = 1
fake = 0
learning_rate = 1e-3
optimD = torch.optim.Adam(modelD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimG = torch.optim.Adam(modelG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

num_epochs = 100


