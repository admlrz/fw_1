import json

import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import datasets, transforms
from datetime import datetime
import sys
from torchsummary import summary
import nuit,mymodule
import os
from PIL import Image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transform = torchvision.transforms.Compose(
        [transforms.Resize(224, 224),
         transforms.ToTensor(),
         transforms.Normalize(mean=(), std=())]
    )

    img_path = ""
    assert os.path.exists(img_path), f'{img_path} do no not exist'
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path),f'{json_path} do not exist'
    with open(json_path,"r") as f:
        class_indict = json.load(f)

    model =






if __name__ == '__main__':
    main()