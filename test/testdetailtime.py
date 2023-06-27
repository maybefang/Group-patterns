import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
import numpy as np

from time import time
# from utils import *
#from models import *
#from trainer import *



import math
from torch.autograd import Variable
from testdetailinference import MaskedConv2d, MaskedMLP,DenseConv2d


__all__ = ['masked_vgg']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class masked_vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(masked_vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.Sequential(
              MaskedMLP(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              MaskedMLP(512, num_classes)
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
        # start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # start.record()
        x = self.feature(x)
        # end.record()
        # torch.cuda.synchronize()
        # #print("masked vgg feature:",start.elapsed_time(end))
        # ft = start.elapsed_time(end)

        # start.record()
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        # end.record()
        # torch.cuda.synchronize()
        # #print("masked vgg pool:",start.elapsed_time(end))
        # pt = start.elapsed_time(end)

        # start.record()
        y = self.classifier(x)
        # end.record()
        # torch.cuda.synchronize()
        # #print("masked linear:",start.elapsed_time(end))
        # ct = start.elapsed_time(end)
        return y#,ft,pt,ct

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                #if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, MaskedMLP):
                m.reset_parameters()

    #mask单独存放使用此函数
    '''
    def init_model(self, masks_dir):#插入mask并在整个net中将weight变为二维
        with open(masks_dir, "rb") as file:
            all_mask_kv = pickle.load(file)  # dic
        
        all_mask_keys = list(all_mask_kv.keys())
        masks_num = len(all_mask_keys)
        i=0
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d):
                layer.mask = all_mask_kv[all_mask_keys[i]]
                layer.reinit_weight()
                i+=1
    '''
    '''
    #mask作为参数在load的时候读入
    def init_model(self):
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d) or isinstance(layer,MaskedMLP):
                layer.reinit_weight()
    '''
    #mask已经存在模型中了，将权重重拼
    def init_model(self):
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d) or isinstance(layer,MaskedMLP):
                layer.reinit_weight()

class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.Sequential(
              nn.Linear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_classes)
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
                conv2d = DenseConv2d(in_channels, v, kernel_size=(3, 3), padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        # start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # start.record()
        x = self.feature(x)
        # end.record()
        # torch.cuda.synchronize()
        # #print("masked vgg feature:",start.elapsed_time(end))
        # ft = start.elapsed_time(end)

        # start.record()
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        # end.record()
        # torch.cuda.synchronize()
        # #print("masked vgg pool:",start.elapsed_time(end))
        # pt = start.elapsed_time(end)

        # start.record()
        y = self.classifier(x)
        # end.record()
        # torch.cuda.synchronize()
        # #print("masked linear:",start.elapsed_time(end))
        # ct = start.elapsed_time(end)
        return y#,ft,pt,ct

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, DenseConv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                #if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.reset_parameters()

    #mask单独存放使用此函数
    '''
    def init_model(self, masks_dir):#插入mask并在整个net中将weight变为二维
        with open(masks_dir, "rb") as file:
            all_mask_kv = pickle.load(file)  # dic
        
        all_mask_keys = list(all_mask_kv.keys())
        masks_num = len(all_mask_keys)
        i=0
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d):
                layer.mask = all_mask_kv[all_mask_keys[i]]
                layer.reinit_weight()
                i+=1
    '''
    '''
    #mask作为参数在load的时候读入
    def init_model(self):
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d) or isinstance(layer,MaskedMLP):
                layer.reinit_weight()
    '''



def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    net = masked_vgg(dataset='cifar10', depth=16)
    # net = vgg(dataset='cifar10', depth=16)

    #model_dir = 'checkpoint/vgg16_5e-3alpha_lr02_bn64/best_keepratio_model.pth'  #25.57
    model_dir = '../checkpoint/vgg16_5e-5alpha_lr02_bn64/best_acc_model.pth'  #50(49.9)
    #model_dir = 'checkpoint/vgg16_1.5e-6alpha_lr02_bn64_bak75/best_acc_model.pth'  #75(73.84)
    
    # model_dir = '../checkpoint/vgg16_nopruning/best_acc_model.pth'
    
    net.load_state_dict(
            torch.load(model_dir, map_location=lambda storage, loc: storage))

    # device = torch.device("cuda")
    # torch.cuda.set_device(args.gpu)

    net.init_model()
    net.to(device)
    
    #for l in net.modules():
    #    if isinstance(l,nn.BatchNorm2d):
    #        print(l.state_dict())

    file_name = 'checkpoint/vgg16_5e-3alpha_lr02_bn64/my_model.pth'
    
    input_data = torch.randn(64,3,32,32).cuda()
    for i in range(1):
        out = net(input_data)
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    rep=1
    all_time=0
    m_time = 0
    d_time = 0
    mft = 0
    mpt = 0
    mct = 0
    at = 0
    bt = 0
    ct = 0
    dt = 0
    et = 0
    ft = 0
    gt = 0
    ht = 0
    jt = 0
    kt = 0
    lt = 0
    mt = 0
    nt = 0
    # for i in range(10):
    #     _,a,b,c,d,e,f,g,h,j,k,l,m,n = net(input_data)

    print()
    print()
    for i in range(rep):
        # start.record()
        # out,ft,pt,ct = net(input_data)
        # #out,a,b,c,d,e,f,g,h,j,k,l,m,n = net(input_data)
        # end.record()
        # torch.cuda.synchronize()
        # m_time += start.elapsed_time(end)
        # mft += ft
        # mpt += pt
        # mct += ct
        # # at += a
        # # bt += b
        # # ct += c
        # # dt += d
        # # et += e
        # # ft += f
        # # gt += g
        # # ht += h
        # # jt += j
        # # kt += k
        # # lt += l
        # # mt += m
        # # nt += n

        start.record()
        out = net(input_data)
        end.record()
        torch.cuda.synchronize()
        d_time += start.elapsed_time(end)
    
    # print("ft:",mft/rep)
    # print("pt:",mpt/rep)
    # print("ct:",mct/rep)
    # print("conv1 :",at)
    # print("conv2 :",bt)
    # print("conv3 :",ct)
    # print("conv4 :",dt)
    # print("conv5 :",et)
    # print("conv6 :",ft)
    # print("conv7 :",gt)
    # print("conv8 :",ht)
    # print("conv9 :",jt)
    # print("conv10:",kt)
    # print("conv11:",lt)
    # print("conv12:",mt)
    # print("conv13:",nt)
    print("mask time:",d_time/rep)
    # print("dense time:",d_time/rep)

    

if __name__ == '__main__':
    # print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main()
