import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict

#Hyperparams
threads = 2 #num_workers
batch = 4
learning_rate = 0.01

data_dir = 'data/cards/'
train_dir = data_dir + 'trainset/'
valid_dir = data_dir + 'evalset/'
test_dir = data_dir + 'testset/'


train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])



validn_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])

test_transforms = transforms.Compose([ transforms.Resize(224),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])


train_data = datasets.ImageFolder(train_dir,
                                transform=train_transforms)

validn_data = datasets.ImageFolder(valid_dir,
                                transform=validn_transforms)

test_data = datasets.ImageFolder(test_dir,
                                transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers = threads)
validnloader = torch.utils.data.DataLoader(validn_data, batch_size=batch, shuffle=True, num_workers = threads)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=True, num_workers = threads)


model = models.vgg16(pretrained=True)
model

for param in model.parameters():
    param.requires_grad = False

resolution = 3*224*224
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(resolution, 10000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(10000, 5000)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(5000, 102)),
                          ('output', nn.LogSoftmax(dim=17))
                          ]))

model.classifier = classifier

classifier


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

epochs = 100
steps = 0
training_loss = 0

for e in range(epochs):
    model.train()
    for images, labels in iter(trainloader):
        steps == 1

        images.resize_(32,3,224,224)

        inputs = Variable(images.cpu())
        targets = Variable(labels.cpu())
        optimizer.zero_grad()

        output = model.forward(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        training_loss += loss.data[0]


        print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(training_loss))

        running_loss = 0