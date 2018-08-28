import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from Net import Net
import torchvision.models as models
from collections import OrderedDict

#Hyperparams and constants
threads = 8 #num_workers
batch = 16
learning_rate = 0.01
img_size = 32

data_dir = 'data/'
train_dir = data_dir + 'trainset/'
valid_dir = data_dir + 'evalset/'
test_dir = data_dir + 'testset/'


train_transforms = transforms.Compose([transforms.Resize(img_size),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(img_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])



validn_transforms = transforms.Compose([transforms.Resize(img_size),
                                        transforms.CenterCrop(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])

test_transforms = transforms.Compose([ transforms.Resize(img_size),
                                       transforms.RandomResizedCrop(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])


train_data = datasets.ImageFolder(train_dir,
                                transform=train_transforms)

#validn_data = datasets.ImageFolder(valid_dir,
#                                transform=validn_transforms)

#test_data = datasets.ImageFolder(test_dir,
#                                transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers = threads)
#validnloader = torch.utils.data.DataLoader(validn_data, batch_size=batch, shuffle=True, num_workers = threads)
#testloader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=True, num_workers = threads)

model = Net(img_size)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)


epochs = 25
steps = 0
loss = 0
if __name__ == '__main__':
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in iter(trainloader):
            steps == 1
            images.resize_(32,3,img_size,img_size)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = loss_func(output, labels)
            loss_func.backward()
            optimizer.step()

            print("Epoch: {}/{}... ".format(epoch + 1, epochs),
                      "Loss: {:.4f}".format(loss))