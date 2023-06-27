import math
import torch
import torch.nn as nn
from torch.autograd import Variable
# from .inference import MaskedConv2d, MaskedMLP
# from .testdetailinference import MaskedConv2d, MaskedMLP, DenseConv2d, DenseMLP


__all__ = ['nop_vgg']


# class BinaryStep(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return (input > 0.).float()

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         zero_index = torch.abs(input) > 1
#         middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
#         additional = 2 - 4 * torch.abs(input)
#         additional[zero_index] = 0.
#         additional[middle_index] = 0.4
#         return grad_input * additional


# # masked linear layer(fc): matrix shape is [outsize, insize]
# # 存了t值，先不用管
# class MaskedMLP(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(MaskedMLP, self).__init__()
#         self.in_size = in_size
#         self.out_size = out_size
#         self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
#         self.bias = nn.Parameter(torch.Tensor(out_size))
#         # self.threshold = nn.Parameter(torch.Tensor(out_size))
#         self.threshold = nn.Parameter(torch.Tensor(1))
#         self.step = BinaryStep.apply
#         self.mask = nn.Parameter(torch.ones([in_size]),requires_grad=False)
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#         with torch.no_grad():
#             # std = self.weight.std()
#             self.threshold.data.fill_(0)

#     #已经插入mask后进行剪枝，此时MLP中的weight形状仍为[cout，cin]
#     def reinit_weight(self):
#         idx = [i.item() for i in self.mask.nonzero()]
#         idx = torch.tensor(idx)
#         masked_weight = torch.index_select(self.weight,1,idx)#重新拼接权重矩阵,实验测出列跳过比行跳过要快
#         self.weight=nn.Parameter(masked_weight.t())#将weight变为[-1，cout]形状
#         self.idx = nn.Parameter(idx,requires_grad=False)


#     def forward(self, input):
#         #abs_weight = torch.abs(self.weight)#[cout,cin]
#         ## threshold = self.threshold.view(abs_weight.shape[0], -1)
#         #mean_weight = torch.sum(abs_weight.data, dim=0) / float(abs_weight.shape[0])
#         #if self.threshold.data < 0:
#         #    self.threshold.data.fill_(0.)
#         #mean_weight = mean_weight - self.threshold  # [cin]
#         #mask = self.step(mean_weight)  # [cin]
#         #ratio = torch.sum(mask) / mask.numel()
#         ############################################################
#         ## print("mlp keep ratio {:.2f}".format(ratio))
#         ## print("mlp threshold {:3f}".format(self.threshold[0]))
#         ###########################################################
#         #if ratio <= 0.01:
#         #    with torch.no_grad():
#         #        # std = self.weight.std()
#         #        self.threshold.data.fill_(0)
#         #    abs_weight = torch.abs(self.weight)
#         #    # threshold = self.threshold.view(abs_weight.shape[0], -1)
#         #    # threshold = self.threshold
#         #    mean_weight = torch.sum(abs_weight.data, dim=0) / float(abs_weight.shape[0])
#         #    mean_weight = mean_weight - self.threshold
#         #    mask = self.step(mean_weight)
#         ##self.mask = mask.repeat(abs_weight.shape[1], 1).t()  # mask=[cin],self.mask=[cin,cout]
#         #self.mask = nn.Parameter(mask,requires_grad=False)#mask不需要更新,mask为1维
#         ## masked_weight = self.weight * mask
#         #masked_weight = torch.einsum('ij,j->ij', self.weight, mask)
#         #output = torch.nn.functional.linear(input, masked_weight, self.bias)
#         # pruned_input = torch.index_select(input,1,self.idx)#由于weight剪枝，input对应列需要进行移除
#         output = torch.mm(input,self.weight.t())
#         return output

# #有mask
# class MaskedConv2d(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
#         super(MaskedConv2d, self).__init__()
#         self.in_channels = in_c
#         self.out_channels = out_c
#         self.kernel_size = kernel_size  # (w,d)
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
        
#         #defin mask
#         self.mask = nn.Parameter(torch.ones([in_c,kernel_size[0],kernel_size[1]]))
#         self.idx = None

#         ## define weight (save)
#         self.weight = nn.Parameter(torch.Tensor(
#             out_c, in_c // groups, *kernel_size
#         ))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_c))
#         else:
#             self.register_parameter('bias', None)
#         # set t, t is a number not a vector
#         self.threshold = nn.Parameter(torch.Tensor(1))  # if out_c=3,init [0,0,0],out_c=2 init [0,0] the numbers are 0  (save)
#         self.step = BinaryStep.apply
#         self.reset_parameters()
#         #self.im2col = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#         with torch.no_grad():
#             self.threshold.data.fill_(0.)

#     #use numpy then to tensor
#     '''
#     def im2col_np(input_data, out_h, out_w, input_shape):
#         N, C, H, W = input_shape
#         #out_h = (H + 2 * pad - ksize) // stride + 1
#         #out_w = (W + 2 * pad - ksize) // stride + 1

#         img = np.pad(input_data, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], "constant")
#         #col = np.zeros((N, C, self.ksize[0], self.ksize[1], out_h, out_w))

#         input_data = input_data.numpy()
#         strides = (*input_data.strides[:-2], input_data.strides[-2]*stride, input_data.strides[-1]*stride, *input_data.strides[-2:])
#         A = as_strided(input_data, shape=(N,C,out_h,out_w,ksize,ksize), strides=strides)
#         col = A.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

#         return col 
#     '''

#     #已经插入mask后，将权重变为二维
#     def reinit_weight(self):
#         weight_shape = self.weight.shape # [out,in,w,h]
#         weight_tile = self.weight.reshape(-1,weight_shape[1]*weight_shape[2]*weight_shape[3]) #reshape the weight to [outchannel,inchannel*h*w]
#         #根据权重提取idx
#         #weight_sum = torch.sum(weight_tile,dim=1)#shape:[outchannel]
#         #idx = [i.item() for i in weight_sum.nonzero()]
#         #根据mask提取idx
#         #self.mask = torch.sum(self.mask,dim=0) #若存储的mask为[out_c,in_c,w,h],所以要把它变为[in_c,w,h]
#         mask = self.mask.reshape(-1)#将mask转换为一维，不然下面一行代码报错
#         self.boolmask = mask.type('torch.BoolTensor')
#         idx = [i.item() for i in mask.nonzero()]
#         idx = torch.tensor(idx)#.cuda()#将idx放在GPU上否则报错
#         #if self.weight.is_cuda():
#         #    idx = idx.cuda()
#         #print("weight_tile.shape:",weight_tile.shape)#[27,64]
#         #print("idx:",idx)
#         masked_weight = torch.index_select(weight_tile,1,idx)#重新拼接权重矩阵
#         self.weight=nn.Parameter(masked_weight.t())#保存weight以[inchannel*kh*kw, outchannel]形状
#         self.idx = nn.Parameter(idx,requires_grad=False)
    
    

#     def im2col(input_data, out_h, out_w, input_shape): 
#         N, C, H, W = input_shape
#         #out_h = (H + 2 * pad - ksize) // stride + 1
#         #out_w = (W + 2 * pad - ksize) // stride + 1
        
#         img = torch.nn.functional.pad(input_data,(self.padding, self.padding, self.padding, self.padding, 0, 0),mode="constant", value=0)

#         strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
#         A = torch.as_strided(img, size=(N,C,out_h,out_w,self.kernel_size[0],self.kernel_size[1]), stride=strides)
#         col = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)

#         return col 


#     def forward(self, x):
#         #weight_shape = self.weight.shape  # [out,in,w,h]
#         ## threshold = self.threshold.view(weight_shape[0], -1)
#         #weight = torch.abs(self.weight)
#         ## weight = weight.view(weight_shape[0], -1)
#         #weight = torch.sum(weight.data, dim=0) / float(weight_shape[0])  # weight.shape=[cin,w,h]
#         #if self.threshold.data < 0:  # make t>=0
#         #    self.threshold.data.fill_(0.)
#         #weight = weight - self.threshold
#         #mask = self.step(weight)  # mask.shape=[cin,w,h]
#         ## mask = mask.view(weight_shape)
#         #ratio = torch.sum(mask) / mask.numel()
#         #####################################################################
#         ## print("conv threshold {:3f}".format(self.threshold[0]))
#         ## print("conv keep ratio {:.2f}".format(ratio))
#         #######################################################################
#         #if ratio <= 0.01:
#         #    with torch.no_grad():
#         #        self.threshold.data.fill_(0.)
#         #    # threshold = self.threshold.view(weight_shape[0], -1)
#         #    weight = torch.abs(self.weight)
#         #    # weight = weight.view(weight_shape[0], -1)
#         #    weight = torch.sum(weight.data, dim=0) / float(weight_shape[0])
#         #    weight = weight - self.threshold
#         #    mask = self.step(weight)
#         #    # mask = mask.view(weight_shape)
#         ## self.mask = mask    #save the [cin,w,h] size mask
#         ##self.mask = mask.repeat(weight_shape[0], 1, 1, 1)  # old:[cin,w,h] new:[cout,cin,w,h]
#         ##masked_weight = torch.einsum('ijkl,jkl->ijkl', self.weight, mask)
#         #weight_tile = self.weight.reshape(-1,weight_shape[1]*weight_shape[2]*weight_shape[3]).t() #reshape the weight to [inchannel*h*w, outchannel]
#         #mask = mask.reshape(-1)#将mask转换为一维，不然下面一行代码报错
#         #idx = [i.item() for i in mask.nonzero()]
#         #idx = torch.tensor(idx).cuda()#将idx放在GPU上否则报错
#         #masked_weight = torch.index_select(weight_tile,0,idx)# spliced into a new matrix
#         #由此以上为用t计算mask并将权重矩阵平铺和重新拼接
#         #conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
#         #                                      padding=self.padding, dilation=self.dilation, groups=self.groups)

#         #to tile the x:
#         x_shape = x.shape
#         N, C, H, W = x_shape
#         k_h,k_w = self.kernel_size
#         out_h = (H + 2 * self.padding - k_h) // self.stride + 1 #conv2d 后3d feature_map的h
#         out_w = (W + 2 * self.padding - k_w) // self.stride + 1
#         #x_tile = torch.from_numpy(im2col(x,out_h,out_w,x_shape)).float()
#         #print("============================================x:",x.shape)
#         #x_tile = self.im2col(x, out_h, out_w, N, C, H, W)
#         ################################   im2col         #########################################
#         img = torch.nn.functional.pad(x,(self.padding, self.padding, self.padding, self.padding, 0, 0),mode="constant", value=0)

#         strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
#         A = torch.as_strided(img, size=(N,C,out_h,out_w,k_h,k_w), stride=strides)
#         #x_tile = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)#之后进行列跳过,网上顺序
#         x_tile = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w,-1)#之后进行列跳过，自己顺序
#         # x_tile = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)#之后进行行跳过

#         # im2col = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
#         # x_tile = im2col(x)
#         # x_tile = x_tile.permute(0,2,1)#[batch_size,kh*kw*cin,xw*xh][64,27,1024]->[64,1024,27]
#         # N, H, W = x_tile.shape
#         # x_tile = x_tile.reshape(N*H,W)
#         ############################################################################################

#         #x_tile = self.im2col(x,out_h,out_w,x_shape)
#         #print("x_tile shape:",x_tile.shape)
#         #boolmask = self.mask.type('torch.BoolTensor')
#         #print("+++++++++++++++++++++++++++",self.boolmask.shape)
#         # x_tile = x_tile[self.boolmask,:].t()#输入重新拼接，此处进行列跳过
#         # x_tile = torch.index_select(x_tile,1,self.idx)  #输入重新拼接，此处进行列跳过
        
#         #print("++++++++++++++++++++++++++++++",x_tile.shape)
#         #conv_out = gemm.mymultiply_nobias(x_tile,masked_weight)
#         #conv_out = gemm.mymultiply_nobias(x_tile,self.weight)
#         w_tile = self.weight.reshape(-1, self.out_channels)
#         conv_out = torch.mm(x_tile, w_tile)
#         #print("conv_out shape:",conv_out.shape)
#         conv_out = conv_out.reshape(N,-1,self.out_channels).permute(0, 2, 1).reshape(N,self.out_channels,out_h,-1) #[batch_size, output_channel(kernel_size[0]), out_h, out_w]
#         #print("after reshape conv_out shape:",conv_out.shape)
#         return conv_out
import math
import torch
import torch.nn as nn
#import gemm

"""
Function for activation binarization
"""


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


# masked linear layer(fc): matrix shape is [outsize, insize]
# 存了t值，先不用管
class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        cudaw = torch.Tensor(out_size, in_size)
        self.weight = nn.Parameter(cudaw)
        cudab = torch.Tensor(out_size)
        self.bias = nn.Parameter(cudab)
        #########################################333
        #self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        #self.bias = nn.Parameter(torch.Tensor(out_size))
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
        idx = [i.item() for i in self.mask.nonzero()]
        idx = torch.tensor(idx)
        masked_weight = torch.index_select(self.weight,1,idx)#重新拼接权重矩阵,实验测出列跳过比行跳过要快
        self.weight=nn.Parameter(masked_weight.t())#将weight变为[-1，cout]形状
        self.idx = nn.Parameter(idx,requires_grad=False)
        self.boolmask = self.mask.type('torch.BoolTensor')
        # print(self.weight.type)
        # print(self.boolmask.type)


    def forward(self, input):
        tstart,tend = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        #pruned_input = torch.index_select(input,1,self.idx)#由于weight剪枝，input对应列需要进行移除
        tstart.record()
        input_t = input.t()

        # start.record()
        pruned_input = input_t[self.boolmask,:]#由于weight剪枝，input对应列需要进行移除
        # end.record()
        # torch.cuda.synchronize()
        # prunint_time = start.elapsed_time(end)

        pruned_input = pruned_input.t()

        # start.record()
        output = torch.mm(pruned_input,self.weight)
        # end.record()
        # torch.cuda.synchronize()
        # mm_time = start.elapsed_time(end)

        tend.record()
        torch.cuda.synchronize()
        t_time = tstart.elapsed_time(tend)
        return output#,t_time#prunint_time,mm_time,t_time

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
        # end.record()
        # torch.cuda.synchronize()
        # mm_time = start.elapsed_time(end)

        tend.record()
        torch.cuda.synchronize()
        t_time = tstart.elapsed_time(tend)
        return output#,t_time#mm_time,t_time

#有mask
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
        cuda_w = torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ).cuda()
        self.weight = nn.Parameter(cuda_w)
        
        #self.weight = nn.Parameter(torch.Tensor(
        #    out_c, in_c // groups, *kernel_size
        #))
        if bias:
            cuda_b = torch.Tensor(out_c).cuda()
            self.bias = nn.Parameter(cuda_b)
            #self.bias = nn.Parameter(torch.Tensor(out_c))
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

    #已经插入mask后，将权重变为二维
    def reinit_weight(self):
        weight_shape = self.weight.shape # [out,in,w,h]
        weight_tile = self.weight.reshape(-1,weight_shape[1]*weight_shape[2]*weight_shape[3]) #reshape the weight to [outchannel,inchannel*h*w]
        #根据权重提取idx
        #weight_sum = torch.sum(weight_tile,dim=1)#shape:[outchannel]
        #idx = [i.item() for i in weight_sum.nonzero()]
        #根据mask提取idx
        #self.mask = torch.sum(self.mask,dim=0) #若存储的mask为[out_c,in_c,w,h],所以要把它变为[in_c,w,h]
        mask = self.mask.reshape(-1)#将mask转换为一维，不然下面一行代码报错
        self.boolmask = mask.type('torch.cuda.BoolTensor')
        idx = [i.item() for i in mask.nonzero()]
        idx = torch.tensor(idx).cuda()#将idx放在GPU上否则报错
        #if self.weight.is_cuda():
        #    idx = idx.cuda()
        #print("weight_tile.shape:",weight_tile.shape)#[27,64]
        #print("idx:",idx)
        masked_weight = torch.index_select(weight_tile,1,idx)#重新拼接权重矩阵
        self.weight=nn.Parameter(masked_weight.t())#保存weight以[inchannel*kh*kw, outchannel]形状
        self.idx = nn.Parameter(idx,requires_grad=False)
        
        #print("weight:",self.weight.device)
        #print("bool mask:",self.boolmask.device)
    
    

    def im2col(self,input_data, out_h, out_w, input_shape): 
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

        strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
        A = torch.as_strided(img, size=(N,C,out_h,out_w,k_h,k_w), stride=strides)
        #x_tile = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)#之后进行列跳过,网上顺序
        #x_tile = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w,-1)#之后进行列跳过，自己顺序
        x_tile = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)#之后进行行跳过

        # im2col = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        # x_tile = im2col(x)
        # x_tile = x_tile.permute(0,2,1)#[batch_size,kh*kw*cin,xw*xh][64,27,1024]->[64,1024,27]
        # N, H, W = x_tile.shape
        # x_tile = x_tile.reshape(N*H,W)
        ############################################################################################
        # end.record()
        # torch.cuda.synchronize()
        # im2col_time = start.elapsed_time(end)
        # print("im2col time in inference.py:",im2col_time)
        # print("x tile masked shape",x_tile.shape)
        

        #x_tile = self.im2col(x,out_h,out_w,x_shape)
        #print("x_tile shape:",x_tile.shape)
        #boolmask = self.mask.type('torch.BoolTensor')
        #print("+++++++++++++++++++++++++++",self.boolmask.shape)
        #x_tile = x_tile[:,self.boolmask]

        # start.record()
        #x_tile = torch.index_select(x_tile,1,self.idx)  #输入重新拼接，此处进行列跳过
        #x_tile = x_tile[:,self.boolmask]  #输入重新拼接，此处进行列跳过
        x_tile = x_tile[self.boolmask,:].t().cuda()  #输入重新拼接，此处进行行跳过
        # end.record()
        # torch.cuda.synchronize()
        # index_select_time = start.elapsed_time(end)

        # start.record()
        #print("x_tile",x_tile.device)
        #print("s weight",self.weight.device)
        conv_out = torch.mm(x_tile,self.weight)
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
        # masked_time = tstart.elapsed_time(tend)
        # print("mask conv time:",masked_time)
        return conv_out#im2col_time,index_select_time,mm_time,reshape_time,masked_time

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

        strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
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

# # masked linear layer(fc): matrix shape is [outsize, insize]
# # 存了t值，先不用管
# class DenseMLP(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(DenseMLP, self).__init__()
#         self.in_size = in_size
#         self.out_size = out_size
#         self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
#         self.bias = nn.Parameter(torch.Tensor(out_size))
#         # self.threshold = nn.Parameter(torch.Tensor(out_size))
#         self.threshold = nn.Parameter(torch.Tensor(1))
#         self.step = BinaryStep.apply
#         self.mask = nn.Parameter(torch.ones([in_size]),requires_grad=False)
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#         with torch.no_grad():
#             # std = self.weight.std()
#             self.threshold.data.fill_(0)

#     def forward(self, input):
#         pruned_input = torch.index_select(input,1,self.idx)#由于weight剪枝，input对应列需要进行移除
#         output = torch.mm(pruned_input,self.weight)
#         return output

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
        x = x.view(x.size(0), -1)
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
    


class nop_masked_vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(nop_masked_vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        flag=1
        
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
            flag=4

        self.classifier = nn.Sequential(
              MaskedMLP(cfg[-1]*flag, 512),
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
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
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
    #mask已经存在模型中了，将权重重拼
    def init_model(self):
        for layer in self.modules():
            if isinstance(layer,MaskedConv2d) or isinstance(layer,MaskedMLP):
                layer.reinit_weight()