import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasets import SiameseMNIST

from model import SiameseMNISTnet

import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train (default: 25)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default:42)')
parser.add_argument('--train_size', type=int, default=1000, help='number of labeled examples')
parser.add_argument('--model_save_path', type=str, default='models_siamese', help='directory to save model')
parser.add_argument('--val_size', type=int, default=1000, help='size of validation set')
args = parser.parse_args()

if not os.path.exists(args.model_save_path):
    os.mkdir(args.model_save_path)

def get_dataloaders(train_set, test_set, train_size, val_size, batch_size):
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(train_set))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]

    train_loader = DataLoader(train_set, pin_memory=True, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(train_set, pin_memory=True, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(test_set, pin_memory=True, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader 

train_dataset = MNIST(root='./data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = MNIST(root='./data', 
                              train=False, 
                              transform=transforms.ToTensor(),
                              download=True)

siamese_train_dataset = SiameseMNIST(train_dataset) 
siamese_test_dataset = SiameseMNIST(test_dataset)

trainloader, valloader, testloader = get_dataloaders(siamese_train_dataset, siamese_test_dataset, args.train_size, args.val_size, args.batch_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = SiameseMNISTnet(10,True)

net = net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

model_name = 'mnist_'+ str(args.train_size)
val_min_acc = 0.0
test_acc = 0.0
flag = False

for epoch in range(args.epochs):  
    net.train()
    running_loss = 0.0
    
    total = 0
    correct = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        labels = labels.float()
        (inputs1, inputs2) = inputs

        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs1,inputs2).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        prob = torch.sigmoid(outputs)
        out = torch.where(prob>=0.5,torch.cuda.FloatTensor(prob.shape).fill_(1),torch.cuda.FloatTensor(prob.shape).fill_(0)).squeeze()
        total += labels.size(0)
        correct += (out == labels).sum().item()
    print('Train: [%d, %5d] loss: %.3f acc: %.3f' % (epoch + 1, args.epochs, running_loss / (i+1),100.0*correct/total))
    running_loss = 0.0
    
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            labels = labels.float()       

            (inputs1, inputs2) = inputs
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            outputs = net(inputs1,inputs2)
            prob = torch.sigmoid(outputs)
            out = torch.where(prob>=0.5,torch.cuda.FloatTensor(prob.shape).fill_(1),torch.cuda.FloatTensor(prob.shape).fill_(0)).squeeze()
            total += labels.size(0)
            correct += (out == labels).sum().item()
            #print(out.shape, labels.shape)
    val_ep_acc = 100*correct/total
    print('Validation Accuracy: %.3f %%' % (val_ep_acc))

    if val_min_acc < val_ep_acc:
        val_min_acc = val_ep_acc
        torch.save(net,args.model_save_path + '/' + model_name + '.pth')
        flag = True

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            labels = labels.float()
            (inputs1, inputs2) = inputs
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            outputs = net(inputs1,inputs2)
            prob = torch.sigmoid(outputs)
            out = torch.where(prob>=0.5,torch.cuda.FloatTensor(prob.shape).fill_(1),torch.cuda.FloatTensor(prob.shape).fill_(0)).squeeze()
            total += labels.size(0)
            correct += (out == labels).sum().item()
    test_ep_acc = 100*correct/total
    print('Test Accuracy: %.3f %%' % (test_ep_acc))
    if(flag):
        test_acc = test_ep_acc
        flag=False

print('Finished training.....')
print('Test accuracy: %.2f %%' % (test_acc))


