import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
import model

import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train (default: 25)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default:42)')
parser.add_argument('--train_size', type=int, default=1000, help='number of labeled examples')
parser.add_argument('--model_save_path', type=str, default='models', help='directory to save model')
parser.add_argument('--val_size', type=int, default=1000, help='size of validation set')
args = parser.parse_args()

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

train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              transform=transforms.ToTensor(),
                              download=True)

trainloader_l, trainloader_u, valloader, testloader = get_dataloaders(train_dataset, test_dataset, args.train_size, args.val_size, args.batch_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = model.resnet18(10,True)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

model_name = 'mnist_pl_'+ str(args.train_size)

alpha = 0.0
val_min_acc = 0
test_min_acc = 0
flag = False

for epoch in range(args.epochs):  

	if(epoch>50 and epoch <=80):
		alpha = (epoch - 50)/30

	net.train()
	running_loss = 0.0
	total = 0
	correct = 0
	for i, data in enumerate(zip(trainloader_l, trainloader_u)):
		inputs_l, labels_l = data[0]
		inputs_u, _ = data[1]
		inputs_l = inputs_l.to(device)
		inputs_u = inputs_u.to(device)
		labels_l = labels_l.to(device)

		optimizer.zero_grad()
		outputs_l, probs_l = net(inputs_l)
		loss = criterion(outputs_l, labels_l)

		outputs_u, probs_u = net(inputs_u)
		_, labels_u = torch.max(probs_u, 1)
		loss += alpha * criterion(outputs_u, labels_u)
		
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
		_, predicted = torch.max(probs_l, 1)

		total += labels_l.size(0)
		correct += (predicted == labels_l).sum().item()

	print('Train: [%d, %5d] loss: %.3f acc: %.3f' % (epoch + 1, args.epochs, running_loss / (i+1),100.0*correct/total))
	running_loss = 0.0

	correct = 0
	total = 0
	net.eval()
	with torch.no_grad():
		for data in valloader:
			images, labels = data
			images = images.to(device)
			labels = labels.to(device)
			outputs, probs = net(images)
			_, predicted = torch.max(probs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	val_ep_acc = 100*correct/total
	print('Validation accuracy: %.3f %%' % (val_ep_acc))

	if val_min_acc <= val_ep_acc:
		val_min_acc = val_ep_acc
		torch.save(net,args.model_save_path + '/' + model_name + '.pth')
		flag = True

	correct = 0
	total = 0
	net.eval()
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			images = images.to(device)
			labels = labels.to(device)
			outputs, probs = net(images)
			_, predicted = torch.max(probs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	test_ep_acc = 100*correct/total
	print('Test accuracy: %.3f %%' % (test_ep_acc))
	if(flag):
		test_min_acc = test_ep_acc
		flag = False

print('Finished training.....')
print('Test accuracy: %.2f %%' % (test_min_acc))
