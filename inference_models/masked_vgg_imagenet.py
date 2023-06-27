import math
import torch
import torch.nn as nn
from torch.autograd import Variable
# from .binarization import MaskedConv2d, MaskedMLP
from .inference import MaskedConv2d, MaskedMLP


__all__ = ['masked_vgg_imagenet']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class masked_vgg_imagenet(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(masked_vgg_imagenet, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        self.avg = nn.AdaptiveAvgPool2d(output_size=(7,7))

        num_classes = 1000

        self.classifier = nn.Sequential(
              MaskedMLP(cfg[-1]*7*7, 4096),
              nn.BatchNorm1d(4096),
              nn.ReLU(inplace=True),
              MaskedMLP(4096, 1000)
            )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=(3, 3), padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, MaskedMLP):
                m.reset_parameters()

    def init_model(self):
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d) or isinstance(layer,MaskedMLP):
                layer.reinit_weight()
