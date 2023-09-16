import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import PIL

class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(self.img_channels, 64, 5, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 5, padding=1)

        self.fc = nn.Linear(512, 1)
