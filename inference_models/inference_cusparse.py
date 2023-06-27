import math
import torch
import torch.nn as nn

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


class MaskedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_size  # (w,d)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        #defin mask
        self.mask = nn.Parameter(torch.ones([in_c,kernel_size[0],kernel_size[1]]))
        self.idx = None

        ## define weight (save)
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        # set t, t is a number not a vector
        self.threshold = nn.Parameter(torch.Tensor(1))  # if out_c=3,init [0,0,0],out_c=2 init [0,0] the numbers are 0  (save)
        self.step = BinaryStep.apply
        self.reset_parameters()
        #self.im2col = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

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

    #已经插入mask后，将权重变为二维
    def reinit_weight(self):
        pass
    
    

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
        N, C, H, W = x.shape
        k_h,k_w = self.kernel_size
        out_h = (H + 2 * self.padding - k_h) // self.stride + 1
        out_w = (W + 2 * self.padding - k_w) // self.stride + 1

        masked_weight = self.weight*self.mask
        sparse_weight = masked_weight.reshape(self.out_channels, -1).to_sparse()
        unfold = torch.nn.Unfold(self.kernel_size,dilation=self.dilation,padding=self.padding,stride=self.stride)
        input_tile = unfold(x)#[batch_size,k*k*cin,out_h*out_w]
        
        input_tile = input_tile.permute(1,0,2).reshape(self.in_channels*self.kernel_size[0]*self.kernel_size[1],-1)
        
        # out = torch.matmul(sparse_weight,input_tile)
        out = torch.sparse.mm(sparse_weight, input_tile)

        out = out.reshape(self.out_channels, N, -1).permute(1,0,2).reshape(N,self.out_channels,out_h,out_w)

        return out

#正常vgg，resnet模型使用的线性层
class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
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

    #已经插入mask后进行剪枝，此时MLP中的weight形状仍为[cout，cin]
    def reinit_weight(self):
        pass


    def forward(self, input):
        masked_weight = self.weight*self.mask
        masked_weight = masked_weight.to_sparse()# 变为系数存储
        input_t = input.t()
        output = torch.mm(masked_weight, input_t).t()
        # output = pruned_input @ self.weight
        return output

#vit模型使用，正常linear层使用上面的MaskedMLP类
# class MaskedMLP(nn.Linear):
#     __constants__ = ['bias']

#     def __init__(self, in_features, out_features):
#         super(MaskedMLP, self).__init__(in_features=in_features,out_features=out_features)
#         self.mask = nn.Parameter(torch.ones([in_features]),requires_grad=False)


#     #def reset_parameters(self):
#     #    init.constant_(self.blocked_mask, 1.0) 

#     def reinit_weight(self):
#         pass

#     def forward(self, input):
#         sparse_weight = self.weight*self.mask
#         sparse_weight = sparse_weight.to_sparse()
#         if len(input.shape)>2:
#             batch,in_h,in_w = input.shape
#             input = input.reshape(batch*in_h,in_w)
#             out = torch.matmul(sparse_weight,input.t()).t()
#             out = out.reshape(batch,-1,self.out_features)
#         else:
#             out = torch.matmul(sparse_weight,input.t()).t()
#         #return F.linear(input, masked_weight, self.bias)
#         return out

#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )