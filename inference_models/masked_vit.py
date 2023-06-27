import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from functools import partial   

# from .inference import MaskedConv2d, MaskedMLP
from .inference_cusparse import MaskedConv2d, MaskedMLP

__all__ = ['masked_vit_cifar_patch4_32', 'masked_vit_tiny_patch16_64']

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
        self.qkv = MaskedMLP(embed_dim, embed_dim*3)
        
        # self.softmax = nn.Softmax(dim = -1) # 对每一行进行softmax
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # Multi-head的拼接，需要一个参数Wo，靠此层进行训练
        self.proj = MaskedMLP(embed_dim, embed_dim)
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
        self.fc1 = MaskedMLP(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = MaskedMLP(hidden_features, out_features)
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
        self.proj = MaskedConv2d(in_c, embed_dim, kernel_size=self.patch_size, stride=self.patch_size[0]) # 进行 patchty 化
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
        self.head = MaskedMLP(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
    
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
        if isinstance(m, MaskedMLP):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def init_model(self):
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d) or isinstance(layer,MaskedMLP):
                layer.reinit_weight()

def masked_vit_cifar_patch4_32(num_classes: int=10):
    model = VisionTransformer(img_size=32,
                              patch_size=2,
                              embed_dim=512,
                              depth=12,
                              heads=16,
                              num_classes=num_classes)
    
    return model

def masked_vit_tiny_patch16_64(num_classes: int=200):
    model = VisionTransformer(img_size=64,
                              patch_size=4,
                              embed_dim=512,
                              depth=12,
                              heads=16,
                              num_classes=num_classes)
    
    return model