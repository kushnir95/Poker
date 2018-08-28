import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision as Tv
class Net(nn.Module):
    def __init__(self, num_classes = 17, img_width_heigh = 32, channels = 1):
        super(Net, self).__init__()

        self.num_classes = num_classes
        self.img_resolution = img_width_heigh**2 * channels
        self.img_side = img_width_heigh
        self.channels = channels

        self.Conv1 = nn.Conv2d(self.channels, 16, 4, 1, 1)
        self.Conv2 = nn.Conv2d(16,16,4,1,1)
        self.Conv3 = nn.Conv2d(16, 32, 4 , 1 ,1)
        self.Conv4 = nn.Conv2d(32,32,4,1,1)

        self.Lin1 = nn.Linear(416,self.num_classes)
        self.SM1 = nn.Softmax(self.num_classes)
        self.Pool = nn.MaxPool2d(kernel_size=2)
        self.ReLU = nn.ReLU()
        self.BN16 = nn.BatchNorm2d(16)
        self.BN32 = nn.BatchNorm2d(32)

    def forward(self, inp):
        out = self.Conv1(inp)
        out = self.BN16(out)
        out = self.ReLU(out)

        out = self.Conv2(out)
        out = self.BN16(out)
        out = self.ReLU(out)

        out = self.Pool(out)

        out = self.Conv3(out)
        out = self.BN32(out)
        out = self.ReLU(out)

        out = self.Conv4(out)
        out = self.BN32(out)
        out = self.ReLU(out)

        out = self.Lin1(out)

        return out

    def num_flat_features(self, x):

        num_features = 1
        for s in x.size()[1:]:# all dimensions except the batch dimension
            num_features*=s
        return num_features


