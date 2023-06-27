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


class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask during training,
    and does real pruning during inference
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
        head_split: int = -1,
        bias_mask: bool = False,
        head_pruning: bool = False,
        row_pruning: bool = True
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Default: ``topK``
            head_split:
                The number of head in the layer. This can also used to make each head prune
                out with same number of rows (so that we can do parallize forward with reshape)
                Default: ``-1`` (means no need for head split)
            bias_mask:
                Prune bias or not
                Default: False
            head_pruning:
                Do Head Pruning or not
                Default: False
            row_pruning:
                Do Row Pruning or Not
                Defualt: True
        """
        super(
            MaskedLinear,
            self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias)

        self.pruning_method = pruning_method
        self.head_split = head_split
        self.bias_mask = bias_mask
        self.head_pruning = head_pruning
        self.row_pruning = row_pruning

        self.inference_mode = False
        # this is used for final block-wise pruning, for init we do not need to
        # worry about that!
        self.block_pruning = False  # We will enable this when needed
        self.block_mask_scores = None  # the mask for block wise pruning
        self.threshold_block = None  # the threshold for block wise pruning

        self.mask_scale = mask_scale
        self.mask_init = mask_init
        if self.row_pruning:
            self.mask_scores = nn.Parameter(
                torch.Tensor(
                    self.weight.size(1),
                    1))  # shape:number of cols * 1
            self.init_mask(self.mask_scores)
            self.threshold_row = nn.Parameter(torch.zeros(1) + 10.0)
        

        self.runsparse = False

    def init_mask(self, mask):
        if self.mask_init == "constant":
            init.constant_(mask, val=self.mask_scale)#使用mask_scale填满mask形状的tensor
        elif self.mask_init == "uniform":
            init.uniform_(mask, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(mask, a=math.sqrt(5))

    def get_mask(self):
        # get head mask
        mask_head = None

        if self.row_pruning:
            mask = TopKBinarizer.apply(
                self.mask_scores, self.threshold_row, -1)
        else:
            mask = None
        return mask_head, mask

    def make_inference_pruning(self, blocksize):
        self.inference_mode = True
        weight_shape = self.weight.size()
        # if there is no block wise pruning needed, we do not have to increase the
        # numner of rows/cols.
        # Otherwise, we need to pad the matrix so that the # of ros/cols is divided by
        # block size
        
        mask_head, mask = self.get_mask()
        # print("mask head:",mask_head)#None

        mask = mask.type('torch.BoolTensor').view(-1)#.to(input.device)
        self.weight = nn.Parameter(self.weight[:, mask])#剪枝的权重存起来

        # we do not need those parameters!
        self.mask_scores = None
        self.head_mask_scores = None
        self.threshold_head = None
        self.threshold_row = None
        self.mask = mask#mask要存起来，之后input重拼接使用
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++",mask.sum()/mask.numel())
        # we need this mask for some Layer O and FC2 pruning
        return mask

    def make_column_purning(self, mask):
        # make column pruning for Layer O and FC2
        self.weight = nn.Parameter(self.weight[:, mask])


    def forward(self, input: torch.tensor):
        if not self.inference_mode:
            output = self.training_forward(input)
        else:
            if not self.block_pruning:
                output = self.inference_forward(input)
            else:
                output = self.block_pruning_forward(input)
        return output


    def inference_forward(self, input: torch.tensor):
        input_prune = input[:,:,self.mask]#对input进行重新拼接
        # print("input pruned:",input_prune.shape)
        # print("pruned weight:",self.weight.shape)
        out = F.linear(input_prune, self.weight, self.bias)#input x weight.t()
        #print("sucessful!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return out

    def training_forward(self, input: torch.tensor):
        mask_head, mask = self.get_mask()

        weight_shape = self.weight.size()
        bias_shape = self.bias.size()
        if self.head_pruning:
            weight_thresholded = (
                self.weight.view(
                    self.head_split, -1) * mask_head).view(weight_shape)
            if self.bias_mask:
                bias_thresholded = (
                    self.bias.view(
                        self.head_split, -1) * mask_head).view(bias_shape)
        else:
            weight_thresholded = self.weight
            bias_thresholded = self.bias
        # Mask weights with computed mask
        if self.row_pruning:
            weight_thresholded = mask * weight_thresholded
            if self.bias_mask:
                bias_thresholded = mask.view(
                    self.bias.size()) * bias_thresholded
            else:
                bias_thresholded = bias_thresholded

        return F.linear(input, weight_thresholded, bias_thresholded)