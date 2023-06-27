import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

# from train_argument import parser, print_args

from time import time
from utils import *
# from inference_models import *
from trainer import *


class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2 - 4 * torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input * additional

class DenseMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(DenseMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        # self.threshold = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(1))
        self.step = BinaryStep.apply
        self.mask = nn.Parameter(torch.ones([in_size]),requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            # std = self.weight.std()
            self.threshold.data.fill_(0)


    def forward(self, input):
        tstart,tend = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        #pruned_input = torch.index_select(input,1,self.idx)#由于weight剪枝，input对应列需要进行移除
        tstart.record()
        weight = self.weight.t()

        # start.record()
        output = torch.mm(input,weight)
        # output = input @ weight
        # end.record()
        # torch.cuda.synchronize()
        # mm_time = start.elapsed_time(end)

        tend.record()
        torch.cuda.synchronize()
        t_time = tstart.elapsed_time(tend)
        return output#,t_time#mm_time,t_time


class DenseConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DenseConv2d, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_size  # (w,d)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        ## define weight (save)
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # with torch.no_grad():
        #     self.threshold.data.fill_(0.)

    #use numpy then to tensor
    '''
    def im2col_np(input_data, out_h, out_w, input_shape):
        N, C, H, W = input_shape
        #out_h = (H + 2 * pad - ksize) // stride + 1
        #out_w = (W + 2 * pad - ksize) // stride + 1

        img = np.pad(input_data, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], "constant")
        #col = np.zeros((N, C, self.ksize[0], self.ksize[1], out_h, out_w))

        input_data = input_data.numpy()
        strides = (*input_data.strides[:-2], input_data.strides[-2]*stride, input_data.strides[-1]*stride, *input_data.strides[-2:])
        A = as_strided(input_data, shape=(N,C,out_h,out_w,ksize,ksize), strides=strides)
        col = A.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

        return col 
    '''

    def im2col(self, input_data, out_h, out_w, input_shape): 
        N, C, H, W = input_shape
        #out_h = (H + 2 * pad - ksize) // stride + 1
        #out_w = (W + 2 * pad - ksize) // stride + 1
        
        img = torch.nn.functional.pad(input_data,(self.padding, self.padding, self.padding, self.padding, 0, 0),mode="constant", value=0)

        strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
        A = torch.as_strided(img, size=(N,C,out_h,out_w,self.kernel_size[0],self.kernel_size[1]), stride=strides)
        col = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)

        return col 


    def forward(self, x):
        #to tile the x:
        # tstart,tend = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # tstart.record()

        x_shape = x.shape
        N, C, H, W = x_shape
        k_h,k_w = self.kernel_size
        out_h = (H + 2 * self.padding - k_h) // self.stride + 1 #conv2d 后3d feature_map的h
        out_w = (W + 2 * self.padding - k_w) // self.stride + 1
        #x_tile = torch.from_numpy(im2col(x,out_h,out_w,x_shape)).float()
        #print("============================================x:",x.shape)
        #x_tile = self.im2col(x, out_h, out_w, N, C, H, W)
        # start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # start.record()
        ################################   im2col         #########################################
        img = torch.nn.functional.pad(x,(self.padding, self.padding, self.padding, self.padding, 0, 0),mode="constant", value=0)

        # strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
        strides = (C*H*W, H*W, W*self.stride, self.stride, H, 1)
        A = torch.as_strided(img, size=(N,C,out_h,out_w,k_h,k_w), stride=strides)
        #x_tile = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)#之后进行列跳过,网上顺序
        x_tile = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w,-1)#之后进行列跳过，自己顺序
        #x_tile = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)#之后进行行跳过

        # im2col = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        # x_tile = im2col(x)
        # x_tile = x_tile.permute(0,2,1)#[batch_size,kh*kw*cin,xw*xh][64,27,1024]->[64,1024,27]
        # N, H, W = x_tile.shape
        # x_tile = x_tile.reshape(N*H,W)
        ############################################################################################
        # end.record()
        # torch.cuda.synchronize()
        # im2col_time = start.elapsed_time(end)
        # print("im2col dense time in inference.py:",im2col_time)
        # print("x tile dense shape",x_tile.shape)

        #x_tile = self.im2col(x,out_h,out_w,x_shape)
        #print("x_tile shape:",x_tile.shape)
        #boolmask = self.mask.type('torch.BoolTensor')
        #print("+++++++++++++++++++++++++++",self.boolmask.shape)
        #x_tile = x_tile[:,self.boolmask]
        #x_tile = torch.index_select(x_tile,1,self.idx)  #输入重新拼接，此处进行列跳过

        # start.record()
        w_tile = self.weight.reshape(-1,self.out_channels)
        # end.record()
        # torch.cuda.synchronize()
        # w_im2col_time = start.elapsed_time(end)

        # start.record()
        conv_out = torch.mm(x_tile, w_tile)
        # end.record()
        # torch.cuda.synchronize()
        # mm_time = start.elapsed_time(end)
        # #print("conv_out shape:",conv_out.shape)

        # start.record()
        conv_out = conv_out.reshape(N,-1,self.out_channels).permute(0, 2, 1).reshape(N,self.out_channels,out_h,-1) #[batch_size, output_channel(kernel_size[0]), out_h, out_w]
        # end.record()
        # torch.cuda.synchronize()
        # reshape_time = start.elapsed_time(end)
        #print("after reshape conv_out shape:",conv_out.shape)
        # tend.record()
        # torch.cuda.synchronize()
        # dense_time = tstart.elapsed_time(tend)
        # print("dense conv time:",dense_time)
        return conv_out#im2col_time, w_im2col_time, mm_time, reshape_time,dense_time


class masked_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, init_weights=True):
        super(masked_BasicBlock, self).__init__()
        self.conv1 = DenseConv2d(in_planes, planes, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DenseConv2d(planes, planes, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                DenseConv2d(in_planes, self.expansion * planes,
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
        self.conv1 = DenseConv2d(in_planes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DenseConv2d(planes, planes, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = DenseConv2d(planes, self.expansion * planes,
                               kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                DenseConv2d(in_planes, self.expansion * planes,
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

        self.conv1 = DenseConv2d(3, 64, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = DenseMLP(512 * block.expansion*flag, num_classes)

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




def ResNet18(dataset="cifar10"):
    return masked_ResNet(masked_BasicBlock, [2, 2, 2, 2], dataset)


def masked_ResNet34(dataset="cifar10"):
    return masked_ResNet(masked_BasicBlock, [3, 4, 6, 3], dataset)


def masked_ResNet50(dataset="cifar10"):
    return masked_ResNet(masked_Bottleneck, [3, 4, 6, 3], dataset)


def masked_ResNet101(dataset="cifar10"):
    return masked_ResNet(masked_Bottleneck, [3, 4, 23, 3], dataset)


def masked_ResNet152(dataset="cifar10"):
    return masked_ResNet(masked_Bottleneck, [3, 8, 36, 3], dataset)

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}
class nop_vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(nop_vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        self.avg = nn.AdaptiveAvgPool2d(output_size=(7,7))

        flag=1
        
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
            flag=4

        self.classifier = nn.Sequential(
              DenseMLP(cfg[-1]*flag, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              DenseMLP(512, num_classes)
            )
        # imagenet
        # self.classifier = nn.Sequential(
        #       DenseMLP(cfg[-1]*49, 4096),
        #       nn.BatchNorm1d(4096),
        #       nn.ReLU(inplace=True),
        #       DenseMLP(4096, 4096),
        #       nn.BatchNorm1d(4096),
        #       nn.ReLU(inplace=True),
        #       DenseMLP(4096, 1000),
        #     )
        # self.DM1 =  DenseMLP(cfg[-1]*flag, 512)
        # self.bn_c = nn.BatchNorm1d(512)
        # self.act = nn.ReLU(inplace=True)
        # self.DM2 = DenseMLP(512, num_classes)
        
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
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        # x = self.avg(x)
        x = x.contiguous().view(x.size(0), -1)
        y = self.classifier(x)
        # x = self.DM1(x)
        # x = self.bn_c(x)
        # x = self.act(x)
        # y = self.DM2(x)

        return y

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

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from functools import partial   

__all__ = ['vit_cifar_patch4_32', 'vit_tiny_patch16_64']

class Attention(nn.Module):
    '''
    Attention Module used to perform self-attention operation allowing the model to attend
    information from different representation subspaces on an input sequence of embeddings.
    The sequence of operations is as follows :-

    Input -> Query, Key, Value -> ReshapeHeads -> Query.TransposedKey -> Softmax -> Dropout
    -> AttentionScores.Value -> ReshapeHeadsBack -> Output
    '''
    def __init__(self, 
                 embed_dim, # 输入token的dim
                 heads=8, 
                 activation=None, 
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // heads  # 每一个head的dim数
        self.scale = head_dim ** -0.5 # ViT-B 就是 768//12 = 64 
        
        # 这里的q,k,v 可以用三个Linear层
        # self.query = nn.Linear(embed_dim, embed_dim)
        # self.key = nn.Linear(embed_dim, embed_dim)
        # self.value = nn.Linear(embed_dim, embed_dim)
        
        # 或者一个Linear层，但是out_channel为三倍，并行的思想
        self.qkv = DenseMLP(embed_dim, embed_dim*3)
        
        # self.softmax = nn.Softmax(dim = -1) # 对每一行进行softmax
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # Multi-head的拼接，需要一个参数Wo，靠此层进行训练
        self.proj = DenseMLP(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # [batch_size, seq_len        , total_embed_dim]
        B, N, C = x.shape
        assert C == self.embed_dim
        
        # 1. qkv -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C//self.heads).permute(2, 0, 3, 1, 4)
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #  # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    '''
    MLP as used in Vision Transformer
    
    Input -> FC1 -> GELU -> Dropout -> FC2 -> Output
    '''
    def __init__(self,
                 in_features,
                 hidden_features = None,
                 out_features = None,
                 act_layer = F.gelu, # 激活函数
                 drop = 0.
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = DenseMLP(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = DenseMLP(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TransformerBlock(nn.Module):
    '''
    Transformer Block combines both the attention module and the feed forward module with layer
    normalization, dropout and residual connections. The sequence of operations is as follows :-
    
    Input -> LayerNorm1 -> Attention -> Residual -> LayerNorm2 -> FeedForward -> Output
      |                                   |  |                                      |
      |-------------Addition--------------|  |---------------Addition---------------|
    '''
    
    def __init__(self, 
                 embed_dim, 
                 heads=8,
                 mlp_ratio=4, # mlp为4倍
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 activation=F.gelu,
                 norm_layer = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, heads=heads, 
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 这里可以选择 drop path， 或者Drop out， 在官方代码中使用了Drop path
        self.drop_path = DropPath(drop_path_ratio)
        # self.drop = nn.Dropout(drop_path_ratio)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=activation, drop=drop_ratio)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        # 原始大小为int，转为tuple，即：img_size原始输入224，变换后为[224,224]
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size) # 16 x 16
        self.img_size = img_size # 224 x 224
        self.patch_size = patch_size # 16 x 16
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 224/16 x 224/16 = 14 x 14
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 14 x 14
        # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
        # 输入维度为3，输出维度为块向量长度
        # 与原文中：分块、展平、全连接降维保持一致
        # 输出为[B, C, H, W]
        self.proj = DenseConv2d(in_c, embed_dim, kernel_size=self.patch_size, stride=self.patch_size[0]) # 进行 patchty 化
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class VisionTransformer(nn.Module):
    '''
    Vision Transformer is the complete end to end model architecture which combines all the above modules
    in a sequential manner. The sequence of the operations is as follows -

    Input -> CreatePatches -> ClassToken, PatchToEmbed , PositionEmbed -> Transformer -> ClassificationHead -> Output
                                   |            | |                |
                                   |---Concat---| |----Addition----|
    '''
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_ratio=4.0,
                 drop_ratio=0.,
                 attn_drop_ratio=0., 
                 drop_path_ratio=0., 
                 embed_layer=PatchEmbed, 
                 norm_layer=None,
                 act_layer=None):
        super().__init__()
        self.name = 'VisionTransformer'
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens =1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or F.gelu
        
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 位置编码 (1,embedim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # 加上类别
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 等差数列的drop_path_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim=embed_dim, heads=heads, mlp_ratio=mlp_ratio,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, activation=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head = DenseMLP(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
    
    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        # x (batch_size, seq_len+1, embed_dim) 
        
        x = self.norm(x)
        return self.pre_logits(x[:, 0]) # batch_size, embed_dim)
    
    def forward(self, x):
        x = self.forward_features(x)
        # (batch_size, embed_dim)
        x = self.head(x)
        # (batch_size, classes)
        
        return x
    
    def _init_vit_weights(m):
        """
        ViT weight initialization
        :param m: module
        """
        if isinstance(m, DenseMLP):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, DenseConv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

def vit_cifar_patch4_32(num_classes: int=10):
    model = VisionTransformer(img_size=32,
                              patch_size=2,
                              embed_dim=512,
                              depth=12,
                              heads=16)
    
    return model

def vit_tiny_patch16_64(num_classes: int=200):
    model = VisionTransformer(img_size=64,
                              patch_size=4,
                              embed_dim=512,
                              depth=12,
                              heads=16)
    
    return model


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def main():
    # save_folder = args.affix
    # data_dir = args.data_root

    # log_folder = os.path.join(args.log_root, save_folder)
    # #log_folder = os.path.join(args.model_root, save_folder,"inference.txt")
    # model_folder = os.path.join(args.model_root, save_folder)
    # #model_folder = os.path.join(args.model_root, save_folder,"best_acc_model.pth")

    # makedirs(log_folder)
    # makedirs(model_folder)

    # setattr(args, 'log_folder', log_folder)
    # setattr(args, 'model_folder', model_folder)

    # logger = create_logger(log_folder, 'inference', 'info')
    # print_args(args, logger)

    device = torch.device("cuda")
    torch.cuda.set_device(7)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    # net = ResNet18(dataset='tiny_imagenet')
    # net = ResNet18()
    net = nop_vgg(dataset='cifar10',depth=16)
    # net = masked_ResNet50(dataset='tiny_imagenet')
    # net = vit_cifar_patch4_32()
    # net = vit_tiny_patch16_64()
    # getModelSize(net)

    # model_dir = os.path.join(model_folder, "best_acc_model.pth")
    # net.load_state_dict(
    #         torch.load(model_dir, map_location=lambda storage, loc: storage))

    #device = torch.device("cuda")
    # torch.cuda.set_device(args.gpu)

    # net.init_model()
    # print(net)
    # for l in net.modules():
    #     print(l,isinstance(l,DenseConv2d))
    net.to(device)

    input_data = torch.randn(32,3,32,32).cuda()
    for i in range(5):
        out = net(input_data)
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    rep=100
    all_time=0
    start.record()
    for i in range(rep):
        # start.record()
        out = net(input_data)
        # end.record()
        # torch.cuda.synchronize()
        # all_time += start.elapsed_time(end)
    end.record()
    torch.cuda.synchronize()
    all_time += start.elapsed_time(end)
    time = all_time/rep
    print(time)



if __name__ == '__main__':
    # args = parser()
    # print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main()
