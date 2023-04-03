import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import datasets, transforms
from datetime import datetime
import sys
from torchsummary import summary
import nuit
import os

class Huochai(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,1,padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,8,3,1,padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2) #size(8,8)
        self.flaten = nn.Flatten()
        self.line1 = nn.Linear(25088, 100)
        self.line2 = nn.Linear(100, num_classes)
        #self.line3 = nn.Linear(100, num_classes)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flaten(x)
        x = self.line1(x)
        x = self.line2(x)
        #x = self.line3(x)
        return x

huochai = Huochai()
print(huochai)
input = torch.ones((64,3,224,224))
output = huochai(input)
print(output.shape)




