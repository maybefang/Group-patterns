import torch
import torch.nn as nn
import torch.nn.functional as F
from .inference import MaskedConv2d, MaskedMLP

__all__ = ['masked_ResNet20']
class masked_ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(masked_ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            MaskedConv2d(inchannel, outchannel, kernel_size=(3,3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            MaskedConv2d(outchannel, outchannel, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                MaskedConv2d(inchannel, outchannel, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class masked_ResNet(nn.Module):
    def __init__(self, masked_ResidualBlock, dataset="cifar10", init_weights=True):
        super(masked_ResNet, self).__init__()
        self.inchannel = 16

        if dataset == "cifar10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        elif dataset == "imagenet":
            num_classes = 1000

        self.conv1 = nn.Sequential(
            MaskedConv2d(3, 16, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(masked_ResidualBlock, 16, 3, stride=1)
        self.layer2 = self.make_layer(masked_ResidualBlock, 32, 3, stride=2)
        self.layer3 = self.make_layer(masked_ResidualBlock, 64, 3, stride=2)
        self.fc = MaskedMLP(64, num_classes)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    #mask作为参数在load的时候读入
    def init_model(self):
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d) or isinstance(layer,MaskedMLP):
                layer.reinit_weight()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    

def masked_ResNet20(dataset="cifar10"):

    return masked_ResNet(masked_ResidualBlock, dataset)
