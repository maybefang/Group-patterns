import math
import torch
import torch.nn as nn
#import gemm

"""
Function for activation binarization
"""


class DenseMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(DenseMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        # self.threshold = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(1))
        # self.step = BinaryStep.apply
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
        # tstart,tend = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        
        # tstart.record()
        # start.record()
        # end.record()
        # torch.cuda.synchronize()
        # _time = start.elapsed_time(end)
        # print("linear __ time:", _time)
        
        start.record()
        weight = self.weight.t()
        

        # start.record()
        output = torch.mm(input,weight)
        end.record()
        torch.cuda.synchronize()
        mm_time = start.elapsed_time(end)
        # print("linear mm time:", mm_time)

        # tend.record()
        # torch.cuda.synchronize()
        # t_time = tstart.elapsed_time(tend)
        # print("linear time:", t_time)
        return output,mm_time#,t_time


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
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # tstart.record()

        start.record()
        x_shape = x.shape
        N, C, H, W = x_shape
        k_h,k_w = self.kernel_size
        out_h = (H + 2 * self.padding - k_h) // self.stride + 1 #conv2d 后3d feature_map的h
        out_w = (W + 2 * self.padding - k_w) // self.stride + 1
        
        # start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # start.record()
        ################################   im2col         #########################################
        img = torch.nn.functional.pad(x,(self.padding, self.padding, self.padding, self.padding, 0, 0),mode="constant", value=0)

        strides = (C*H*W, H*W, W*self.stride, self.stride, H, 1)
        A = torch.as_strided(img, size=(N,C,out_h,out_w,k_h,k_w), stride=strides)
        #x_tile = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)#之后进行列跳过,网上顺序
        x_tile = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w,-1)#之后进行列跳过，自己顺序
        #x_tile = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)#之后进行行跳过

        ############################################################################################
        end.record()
        torch.cuda.synchronize()
        im2col_time = start.elapsed_time(end)
        # print("im2col time:",im2col_time)
        
        start.record()
        w_tile = self.weight.reshape(-1,self.out_channels)
        end.record()
        torch.cuda.synchronize()
        w_im2col_time = start.elapsed_time(end)
        # print("w_tile time:",w_im2col_time)

        start.record()
        conv_out = torch.mm(x_tile, w_tile)
        end.record()
        torch.cuda.synchronize()
        mm_time = start.elapsed_time(end)
        # print("mm time:",mm_time)

        start.record()
        conv_out = conv_out.reshape(N,-1,self.out_channels).permute(0, 2, 1).reshape(N,self.out_channels,out_h,-1) #[batch_size, output_channel(kernel_size[0]), out_h, out_w]
        end.record()
        torch.cuda.synchronize()
        reshape_time = start.elapsed_time(end)
        # print("convout reshape time:",reshape_time)

        # tend.record()
        # torch.cuda.synchronize()
        # dense_time = tstart.elapsed_time(tend)
        # print("conv time:",dense_time)
        return conv_out, im2col_time, w_im2col_time, mm_time, reshape_time#,dense_time



class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        # self.threshold = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(1))
        # self.step = BinaryStep.apply
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
        # for i in self.mask.nonzero():
        #     print(i)
        idx = [i.item() for i in self.mask.nonzero()]
        idx = torch.tensor(idx)
        masked_weight = torch.index_select(self.weight,1,idx)#重新拼接权重矩阵,实验测出列跳过比行跳过要快
        self.weight=nn.Parameter(masked_weight.t())#将weight变为[-1，cout]形状
        self.idx = nn.Parameter(idx,requires_grad=False)
        self.boolmask = self.mask.type('torch.BoolTensor')


    def forward(self, input):
        # tstart,tend = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # tstart.record()

        start.record()
        # input_t = input.t()
        # pruned_input = input_t[self.boolmask, :].t()#由于weight剪枝，input对应列需要进行移除
        pruned_input = torch.index_select(input,1,self.idx)#.t()
        # pruned_input = input[:,self.boolmask]
        end.record()
        torch.cuda.synchronize()
        prune_input_time = start.elapsed_time(end)
        # print("masked prune input time:",prune_input_time)

        start.record()
        output = torch.mm(pruned_input,self.weight)
        end.record()
        torch.cuda.synchronize()
        mm_time = start.elapsed_time(end)
        # print("masked mm time:",mm_time)

        # tend.record()
        # torch.cuda.synchronize()
        # dense_time = tstart.elapsed_time(tend)
        # print("masked linear time:",dense_time)
        return output, prune_input_time, mm_time

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
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        # set t, t is a number not a vector
        self.threshold = nn.Parameter(torch.Tensor(1))  # if out_c=3,init [0,0,0],out_c=2 init [0,0] the numbers are 0  (save)
        # self.step = BinaryStep.apply
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
        weight_shape = self.weight.shape # [out,in,w,h]
        weight_tile = self.weight.reshape(-1,weight_shape[1]*weight_shape[2]*weight_shape[3]) #reshape the weight to [outchannel,inchannel*h*w]
        #根据权重提取idx
        #weight_sum = torch.sum(weight_tile,dim=1)#shape:[outchannel]
        #idx = [i.item() for i in weight_sum.nonzero()]
        #根据mask提取idx
        #self.mask = torch.sum(self.mask,dim=0) #若存储的mask为[out_c,in_c,w,h],所以要把它变为[in_c,w,h]
        mask = self.mask.reshape(-1)#将mask转换为一维，不然下面一行代码报错
        self.boolmask = mask.type('torch.BoolTensor')
        idx = [i.item() for i in mask.nonzero()]
        idx = torch.tensor(idx)#.cuda()#将idx放在GPU上否则报错
        #if self.weight.is_cuda():
        #    idx = idx.cuda()
        #print("weight_tile.shape:",weight_tile.shape)#[27,64]
        #print("idx:",idx)
        masked_weight = torch.index_select(weight_tile,1,idx)#重新拼接权重矩阵
        self.weight=nn.Parameter(masked_weight.t())#保存weight以[inchannel*kh*kw, outchannel]形状
        self.idx = nn.Parameter(idx,requires_grad=False)
    
    

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
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # tstart.record()
        
        start.record()
        x_shape = x.shape
        N, C, H, W = x_shape
        k_h,k_w = self.kernel_size
        out_h = (H + 2 * self.padding - k_h) // self.stride + 1 #conv2d 后3d feature_map的h
        out_w = (W + 2 * self.padding - k_w) // self.stride + 1
        
        ################################   im2col         #########################################
        img = torch.nn.functional.pad(x,(self.padding, self.padding, self.padding, self.padding, 0, 0),mode="constant", value=0)
        
        strides = (C*H*W, H*W, W*self.stride, self.stride, H, 1)
        A = torch.as_strided(img, size=(N,C,out_h,out_w,k_h,k_w), stride=strides)
        #x_tile = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)#之后进行列跳过,网上顺序
        # x_tile = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w,-1)#之后进行列跳过，自己顺序
        x_tile = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)#之后进行行跳过

        ############################################################################################
        end.record()
        torch.cuda.synchronize()
        im2col_time = start.elapsed_time(end)
        # print("masked im2col time:",im2col_time)
        
        # start.record()
        # end.record()
        # torch.cuda.synchronize()
        # _time = start.elapsed_time(end)
        # print("masked __ time:",_time)

        start.record()
        x_tile = x_tile[self.boolmask,:].t()#输入重新拼接，此处进行行跳过
        # x_tile = torch.index_select(x_tile,0,self.idx).t()  #输入重新拼接，此处进行列跳过
        end.record()
        torch.cuda.synchronize()
        prune_input_time = start.elapsed_time(end)
        # print("masked prune input time:",prune_input_time)

        start.record()
        conv_out = torch.mm(x_tile,self.weight)
        
        end.record()
        torch.cuda.synchronize()
        mm_time = start.elapsed_time(end)
        # print("masked mm time:",mm_time)

        start.record()
        conv_out = conv_out.reshape(N,-1,self.out_channels).permute(0, 2, 1).reshape(N,self.out_channels,out_h,-1) #[batch_size, output_channel(kernel_size[0]), out_h, out_w]
        end.record()
        torch.cuda.synchronize()
        reshape_time = start.elapsed_time(end)
        # print("masked convout reshape time:",reshape_time)
        # tend.record()
        # torch.cuda.synchronize()
        # dense_time = tstart.elapsed_time(tend)
        # print("masked conv time:",dense_time)
        
        return conv_out, im2col_time, prune_input_time, mm_time, reshape_time
