import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from .inference import MaskedConv2d, MaskedMLP
from .inference_cusparse import MaskedConv2d, MaskedMLP
from .inference_2015 import MaskedConv2d2015

__all__ = ['masked_ResNet18', 'masked_ResNet18_2015', 'masked_ResNet34', 'masked_ResNet50', 'masked_ResNet101', 'masked_ResNet152']
# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class masked_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, init_weights=True):
        super(masked_BasicBlock, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion * planes,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class masked_BasicBlock2015(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, init_weights=True):
        super(masked_BasicBlock2015, self).__init__()
        self.conv1 = MaskedConv2d2015(in_planes, planes, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d2015(planes, planes, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d2015(in_planes, self.expansion * planes,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class masked_Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(masked_Bottleneck, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = MaskedConv2d(planes, self.expansion * planes,
                               kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion * planes,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class masked_ResNet(nn.Module):
    def __init__(self, block, num_blocks, dataset="cifar10"):
        super(masked_ResNet, self).__init__()
        self.in_planes = 64

        flag=1

        if dataset == "cifar10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        elif dataset == "tiny_imagenet":
            num_classes = 200
            flag=4

        self.conv1 = MaskedConv2d(3, 64, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = MaskedMLP(512 * block.expansion*flag, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    #mask作为参数在load的时候读入
    def init_model(self):
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d) or isinstance(layer,MaskedMLP):
                layer.reinit_weight()
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)#4:32*32,28:224*224
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class masked_ResNet2015(nn.Module):
    def __init__(self, block, num_blocks, dataset="cifar10"):
        super(masked_ResNet2015, self).__init__()
        self.in_planes = 64

        flag=1

        if dataset == "cifar10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        elif dataset == "tiny_imagenet":
            num_classes = 200
            flag=4

        self.conv1 = MaskedConv2d2015(3, 64, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion*flag, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    #mask作为参数在load的时候读入
    def init_model(self):
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d2015):
                layer.reinit_weight()
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)#4:32*32,28:224*224
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




def masked_ResNet18(dataset="cifar10"):
    return masked_ResNet(masked_BasicBlock, [2, 2, 2, 2], dataset)

def masked_ResNet18_2015(dataset="cifar10"):
    return masked_ResNet2015(masked_BasicBlock2015, [2, 2, 2, 2], dataset)


def masked_ResNet34(dataset="cifar10"):
    return masked_ResNet(masked_BasicBlock, [3, 4, 6, 3], dataset)


def masked_ResNet50(dataset="cifar10"):
    return masked_ResNet(masked_Bottleneck, [3, 4, 6, 3], dataset)


def masked_ResNet101(dataset="cifar10"):
    return masked_ResNet(masked_Bottleneck, [3, 4, 23, 3], dataset)


def masked_ResNet152(dataset="cifar10"):
    return masked_ResNet(masked_Bottleneck, [3, 8, 36, 3], dataset)

'''
def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
'''