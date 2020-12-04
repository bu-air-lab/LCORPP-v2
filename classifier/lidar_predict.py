# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 23:42:21 2020

@author: cckklt
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image,ImageOps

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        #1 input image channel, 32 output channel, 3*3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        
        #32 input channels,64 output channedl, 3*3 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        #nn.Dropout2d() will help promote independence between feature maps and should be used instead.
        #torch.nn.Dropout2d
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        #apply a linear transformation to the incoming data:y=xA+b,torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        #print("Size after conv1:",x.size())
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        #print("Size after cov2:",x.size())
        x = F.max_pool2d(x, 2)

        #print("Size after pooling:",x.size())
        x = self.dropout1(x)
        
        #Flattens a contiguous range of dims in a tensor
        x = torch.flatten(x, 1)
        #print("Size after flattern:", x.size())
                
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def image_loader(transform,image_name):
    image = Image.open(image_name)
    image = transform(image).float()
    image = torch.as_tensor(image)
    image = image.unsqueeze(0)
    return image

def predict(image, model):
    with torch.no_grad():
        out = model(image)
        _, predicted = torch.max(out, 1)
        predicted = predicted.numpy()[0]
    return predicted

device = torch.device("cpu")

transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
model=Net().to(device)
model.load_state_dict(torch.load("lidar_cnn.pt"))
model.eval()

image=image_loader(transform,"013374.jpg")
#predict=model(image.detach())
print("The prediction is:",predict(image,model))
