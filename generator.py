import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()
        self.z = torch.normal(0, 1, size=(1, 100)) # sample from normal distribution instead of uniform