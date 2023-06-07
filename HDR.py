import torch
import torch.nn as nn
# import torch.optim as optim
# import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor, Grayscale
# from torch.utils.data import random_split

import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import time
# from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image


# Data agnostic
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

### MODEL ###

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10


# Model Class
class LeNet(nn.Module):
    def __init__(self, channels, classes):
        super(LeNet,self).__init__()
        
        # LeNet Architecture
        
        # First Block
        self.convBlock1 = nn.Conv2d(channels, 20, (5,5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))

        # Second Block
        self.convBlock2 = nn.Conv2d(20,50, (5,5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2,2), (2,2))

        # Fully Connected Layer
        self.fullyConnected1 = nn.Linear(800,500)
        self.relu3 = nn.ReLU()

        # Softmax for logit to predictions
        self.fullyConnected2 = nn.Linear(500,classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self,x):

        # First Block Pass
        x = self.maxpool1(self.relu1(self.convBlock1(x)))
        
        # Second Block Pass
        x = self.maxpool2(self.relu2(self.convBlock2(x)))
        
        # Flatten and pass to Fully Connected Layer
        x = self.relu3(self.fullyConnected1(torch.flatten(x,1)))

        # Softmax Pass
        x = self.fullyConnected2(x)
        output = self.logSoftmax(x)

        return output





# model = LeNet(1,10)
# state_dict = torch.load("HDR.pth", torch.device(device))
# model.load_state_dict(state_dict)
# model = model.to(device)

def HDR(x):

    res = "LES GO"

    model = LeNet(1,10)
    state_dict = torch.load("HDR.pth", torch.device(device))
    model.load_state_dict(state_dict)
    model = model.to(device)


    print("CALLED HDR")
    x = x.split(",")[1]
    # print(x)

    x = base64.b64decode(x)
    image = Image.open(BytesIO(x))
    desiredSize = (28,28)

    # Transform
    transform = transforms.Compose([
        Resize(desiredSize),
        transforms.Grayscale(),
        ToTensor(),
    ])
    transformed_image = transform(image)


    # print(transformed_image.shape)

    numpy_image = transformed_image.numpy().squeeze()
    normalized_image = (numpy_image - numpy_image.min()) / (numpy_image.max() - numpy_image.min()) * 255
    uint8_image = normalized_image.astype(np.uint8)

    imageSave = Image.fromarray(uint8_image)
    imageSave.save("USERDIGIT.png")

    # a = transformed_image.numpy()
    # a = a.transpose(1,2,0)
    # plt.imshow(a, cmap="gray")
    # plt.axis("off")

    model.eval()
    with torch.no_grad():
        res = torch.argmax(model(transformed_image.unsqueeze(0).to(device))).item()
    print(f'RESULT: {res}')

    # del model

    return res

# HDR("a")


