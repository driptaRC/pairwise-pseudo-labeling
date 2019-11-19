import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasets import SiameseMNIST

from model import SiameseNet

import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 25)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default:42)')
parser.add_argument('--train_size', type=int, default=1000, help='number of labeled examples')
parser.add_argument('--model_save_path', type=str, default='models_siamese', help='directory to save model')
parser.add_argument('--val_size', type=int, default=1000, help='size of validation set')
parser.add_argument('--T1', type=int, default=50, help='start epoch of pseudo-labels')
parser.add_argument('--T2', type=int, default=80, help='end epoch of pseudo-labels')

args = parser.parse_args()

if not os.path.exists(args.model_save_path):
    os.mkdir(args.model_save_path)

def get_dataloaders(train_set, test_set, train_size, val_size, batch_size):
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(train_set))
    labeled_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    unlabeled_indices = indices[train_size+val_size:]
    batch_size_l = int(0.4*batch_size)
    batch_size_u = batch_size - batch_size_l
    train_loader_l = DataLoader(train_set, pin_memory=True, batch_size=batch_size_l, sampler=SubsetRandomSampler(labeled_indices))
    train_loader_u = DataLoader(train_set, pin_memory=True, batch_size=batch_size_u, sampler=SubsetRandomSampler(unlabeled_indices))
    val_loader = DataLoader(train_set, pin_memory=True, batch_size=batch_size_u, sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(test_set, pin_memory=True, batch_size=batch_size)
    
    return train_loader_l, train_loader_u, val_loader, test_loader 

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

trainloader_l, trainloader_u, valloader, testloader = get_dataloaders(siamese_train_dataset, siamese_test_dataset, args.train_size, args.val_size, args.batch_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = SiameseNet(10,True)

net = net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

model_name = 'mnist_pl'+ str(args.train_size)
alpha = 0.0
val_min_acc = 0.0
test_acc = 0.0
flag = False

for epoch in range(args.epochs):  
    if(epoch>args.T1 and epoch <=args.T2):
        alpha = (epoch - args.T1)/(args.T2-args.T1)

    net.train()
    running_loss = 0.0
    
    total = 0
    correct = 0

    for i, data in enumerate(zip(trainloader_l, trainloader_u)):
        inputs_l, labels_l = data[0]
        labels_l = labels_l.float()
        inputs_u, _ = data[1]
        (inputs_l_1, inputs_l_2) = inputs_l
        (inputs_u_1, inputs_u_2) = inputs_u

        inputs_l_1 = inputs_l_1.to(device)
        inputs_l_2 = inputs_l_2.to(device)
        inputs_u_1 = inputs_u_1.to(device)
        inputs_u_2 = inputs_u_2.to(device)
        labels_l = labels_l.to(device)

        optimizer.zero_grad()

        outputs_l = net(inputs_l_1,inputs_l_2).squeeze()
        loss = criterion(outputs_l, labels_l)

        outputs_u = net(inputs_u_1,inputs_u_2).squeeze()
        proba = torch.sigmoid(outputs_u)
        labels_u = torch.where(proba>=0.6,torch.cuda.FloatTensor(proba.shape).fill_(1),torch.cuda.FloatTensor(proba.shape).fill_(0)).squeeze()
        loss += alpha * criterion(outputs_u, labels_u)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        prob = torch.sigmoid(outputs_l)
        out = torch.where(prob>=0.5,torch.cuda.FloatTensor(prob.shape).fill_(1),torch.cuda.FloatTensor(prob.shape).fill_(0)).squeeze()
        total += labels_l.size(0)
        correct += (out == labels_l).sum().item()
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

    if val_min_acc < val_ep_acc and epoch>args.T2: 
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


