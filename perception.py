import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import subprocess
from pathlib import Path
import random

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

def select_train_images(ins_list,em_data_path='lidar_converted_data/class_A',cr_data_path="lidar_converted_data/class_B",em_destination='lidar_train_data/class_A',cr_destination='lidar_train_data/class_B'):
    num_cr=0
    num_em=0
    for ins in ins_list:
        if ins['cr']=="crowded":
            num_cr+=1
        elif ins['cr']=="empty":
            num_em+=1
                  
    cr_image_list=os.listdir(cr_data_path)
    #print(cr_image_list)
    selected_cr_image_list=random.sample(cr_image_list,k=num_cr)
            
    em_image_list=os.listdir(em_data_path)
    #print(em_image_list)
    selected_em_image_list=random.sample(em_image_list,k=num_em)
            
    for f1 in selected_cr_image_list:
        shutil.copy(cr_data_path+"/"+f1,cr_destination)
    for f2 in selected_em_image_list:
        shutil.copy(em_data_path+"/"+f2,em_destination)
            
    print("Lidar train data is generated")
      
def select_test_images(em_data_path='lidar_converted_data/class_A',cr_data_path="lidar_converted_data/class_B",em_destination='lidar_test_data/class_A',cr_destination='lidar_test_data/class_B'):
    cr_image_list=os.listdir(cr_data_path)
    em_image_list=os.listdir(em_data_path)
    selected_em_image_list=random.sample(em_image_list,k=50)
    selected_cr_image_list=random.sample(cr_image_list,k=50)
            
    for f1 in selected_em_image_list:
        shutil.move(em_data_path+"/"+f1,em_destination)
            
    for f2 in selected_cr_image_list:
        shutil.move(cr_data_path+"/"+f2,cr_destination)

    print("Lidar test data is generated")

def delete_train_images():
    shutil.rmtree("lidar_train_data/class_A")
    shutil.rmtree("lidar_train_data/class_B")
    os.makedirs("lidar_train_data/class_A")
    os.makedirs("lidar_train_data/class_B")
            
def delete_test_images():
    shutil.rmtree("lidar_test_data/class_A")
    shutil.rmtree("lidar_test_data/class_B")
    os.makedirs("lidar_test_data/class_A")
    os.makedirs("lidar_test_data/class_B")
