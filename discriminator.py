import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import PIL

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 5, padding='same')
        self.bnorm = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)

        x = self.pool(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels

        self.conv1 = Block(img_channels, 64)
        self.conv2 = Block(64, 128)
        self.conv3 = Block(128, 256)
        self.conv4 = Block(256, 512)
        

        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = nn.Flatten()(x)
        x = self.fc(x)
        
        return x
