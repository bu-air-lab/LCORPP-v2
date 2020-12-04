# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:01:03 2020

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        #1 input image channel, 32 output channel, 3*3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        
        #32 input channels,64 output channedl, 3*3 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        #nn.Dropout2d() will help promote independence between feature maps and should be used instead.
        #torch.nn.Dropout2d
        #self.dropout1 = nn.Dropout2d(0.25)
        #self.dropout2 = nn.Dropout2d(0.5)
        
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
        #x = self.dropout1(x)
        
        #Flattens a contiguous range of dims in a tensor
        x = torch.flatten(x, 1)
        #print("Size after flattern:", x.size())
                
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    #Get a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
    conf_matrix=torch.zeros(2,2)
    #tensor.view()
    #Returns a new tensor with the same data as the self tensor but of a different shape.
    #The returned tensor shares the same data and must have the same number of elements, but may have a different size.
    with torch.no_grad():
        for data, target in test_loader:
            #print("data,target",data,target)
            data, target = data.to(device), target.to(device)
            #print("data,target to device",data,target)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print("pred is",pred)
            for t, p in zip(target.view(-1), pred.view(-1)):
                conf_matrix[t.long(), p.long()] += 1
            correct += pred.eq(target.view_as(pred)).sum().item()
        print("confusion matrix is:", conf_matrix)

        #torch.diag()
        #If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.
        #If input is a matrix (2-D tensor), then returns a 1-D tensor with the diagonal elements of input.
        print(conf_matrix.diag()/conf_matrix.sum(1))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return conf_matrix.diag()/conf_matrix.sum(1)

def perceive_learn():
    #training settings
    parser = argparse.ArgumentParser(description='Training Settings')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
      
    device = torch.device("cpu")
    #A torch.device is an object representing the device on which a torch.Tensor is or will be allocated.
      
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_data=datasets.ImageFolder("lidar_train_data",transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=4,
                                          shuffle=True)
    test_data=datasets.ImageFolder("lidar_test_data",transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=4,
                                          shuffle=True)
      
    model=Net().to(device)
    print(model)
      
    #To use torch.optim you have to construct an optimizer object, that will hold the current state 
    #and will update the parameters based on the computed gradients.
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
      
      
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        conf_matrix=test(model, device, test_loader).tolist()
        scheduler.step()
    #state_dict(),returns a dictionary containing a whole state of the model
    #print("conf_matrix is",conf_matrix)
    torch.save(model.state_dict(), "lidar_cnn0922.pt")
    return conf_matrix



def main():
    perceive_learn()
      



      
      
if __name__ == '__main__':
    main()