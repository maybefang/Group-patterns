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


class MaskedConv2d2015(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d2015, self).__init__()
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
        weight_shape = self.weight.shape # [out,in,w,h]
        weight_tile = self.weight.reshape(-1,weight_shape[1]*weight_shape[2]*weight_shape[3]) #reshape the weight to [outchannel,inchannel*h*w]

        mask = self.mask.reshape(-1)#将mask转换为一维，不然下面一行代码报错
        self.boolmask = mask.type('torch.BoolTensor')
        idx = [i.item() for i in mask.nonzero()]
        idx = torch.tensor(idx)#.cuda()#将idx放在GPU上否则报错

        # masked_weight = torch.index_select(weight_tile,1,idx)#重新拼接权重矩阵
        masked_weight = weight_tile[:, self.boolmask]
        masked_weight = masked_weight.t()#保存weight以[inchannel*kh*kw, outchannel]形状
        self.idx = nn.Parameter(idx,requires_grad=False)

        #to tile the x:
        x_shape = x.shape
        N, C, H, W = x_shape
        k_h,k_w = self.kernel_size
        out_h = (H + 2 * self.padding - k_h) // self.stride + 1 #conv2d 后3d feature_map的h
        out_w = (W + 2 * self.padding - k_w) // self.stride + 1
        #x_tile = torch.from_numpy(im2col(x,out_h,out_w,x_shape)).float()
        #print("============================================x:",x.shape)
        #x_tile = self.im2col(x, out_h, out_w, N, C, H, W)
        ################################   im2col         #########################################
        img = torch.nn.functional.pad(x,(self.padding, self.padding, self.padding, self.padding, 0, 0),mode="constant", value=0)
        # strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
        strides = (C*H*W, H*W, W*self.stride, self.stride, H, 1)
        A = torch.as_strided(img, size=(N,C,out_h,out_w,k_h,k_w), stride=strides)
        #x_tile = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)#之后进行列跳过,网上顺序
        # x_tile = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w,-1)#之后进行列跳过，自己顺序
        x_tile = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)#之后进行行跳过

        # im2col = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        # x_tile = im2col(x)
        # x_tile = x_tile.permute(0,2,1)#[batch_size,kh*kw*cin,xw*xh][64,27,1024]->[64,1024,27]
        # N, H, W = x_tile.shape
        # x_tile = x_tile.reshape(N*H,W)
        ############################################################################################

        #x_tile = self.im2col(x,out_h,out_w,x_shape)
        #print("x_tile shape:",x_tile.shape)
        #boolmask = self.mask.type('torch.BoolTensor')
        # print("+++++++++++++++++++++++++++",self.boolmask.shape)
        # print("+++++++++++++++++++++++++++in maksed conv2d:",type(x_tile))
        x_tile = x_tile[self.boolmask, :].t()#输入重新拼接，此处进行列跳过
        # x_tile = torch.index_select(x_tile,1,self.idx)  #输入重新拼接，此处进行列跳过
        
        # print("++++++++++++++++++++++++++++++",x_tile.shape)
        #conv_out = gemm.mymultiply_nobias(x_tile,masked_weight)
        #conv_out = gemm.mymultiply_nobias(x_tile,self.weight)
        conv_out = torch.mm(x_tile, masked_weight)
        #print("conv_out shape:",conv_out.shape)
        conv_out = conv_out.reshape(N,-1,self.out_channels).permute(0, 2, 1).reshape(N,self.out_channels,out_h,-1) #[batch_size, output_channel(kernel_size[0]), out_h, out_w]
        #print("after reshape conv_out shape:",conv_out.shape)
        return conv_out

# class MaskedConv2d2015(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
#         super(MaskedConv2d2015, self).__init__()
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
        
#         pass
    
    

#     def im2col(self, input_data, out_h, out_w, input_shape): 
#         N, C, H, W = input_shape
#         #out_h = (H + 2 * pad - ksize) // stride + 1
#         #out_w = (W + 2 * pad - ksize) // stride + 1
        
#         img = torch.nn.functional.pad(input_data,(self.padding, self.padding, self.padding, self.padding, 0, 0),mode="constant", value=0)

#         strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
#         A = torch.as_strided(img, size=(N,C,out_h,out_w,self.kernel_size[0],self.kernel_size[1]), stride=strides)
#         col = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)

#         return col 


#     def forward(self, x):

#         weight_shape = self.weight.shape # [out,in,w,h]
#         weight_tile = self.weight.reshape(-1,weight_shape[1]*weight_shape[2]*weight_shape[3]) #reshape the weight to [outchannel,inchannel*h*w]
#         mask = self.mask.reshape(-1)#将mask转换为一维，不然下面一行代码报错
#         self.boolmask = mask.type('torch.BoolTensor')
#         masked_weight = weight_tile[:,self.boolmask]
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
#         # strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
#         strides = (C*H*W, H*W, W*self.stride, self.stride, H, 1)
#         A = torch.as_strided(img, size=(N,C,out_h,out_w,k_h,k_w), stride=strides)
#         # x_tile = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)#之后进行列跳过,网上顺序
#         # x_tile = A.permute(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w,-1)#之后进行列跳过，自己顺序
#         x_tile = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)#之后进行行跳过

#         # im2col = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
#         # x_tile = im2col(x)
#         # x_tile = x_tile.permute(0,2,1)#[batch_size,kh*kw*cin,xw*xh][64,27,1024]->[64,1024,27]
#         # N, H, W = x_tile.shape
#         # x_tile = x_tile.reshape(N*H,W)
#         ############################################################################################

#         #x_tile = self.im2col(x,out_h,out_w,x_shape)
#         #print("x_tile shape:",x_tile.shape)
#         #boolmask = self.mask.type('torch.BoolTensor')
#         # print("+++++++++++++++++++++++++++",self.boolmask.shape)
#         # print("+++++++++++++++++++++++++++in maksed conv2d:",type(x_tile))
#         # x_tile = x_tile[:, self.boolmask].t()#输入重新拼接，此处进行列跳过
#         x_tile = x_tile[self.boolmask, :]#行跳过
#         # x_tile = torch.index_select(x_tile,1,self.idx)  #输入重新拼接，此处进行列跳过
        
#         # print("++++++++++++++++++++++++++++++",x_tile.shape)
#         #conv_out = gemm.mymultiply_nobias(x_tile,masked_weight)
#         #conv_out = gemm.mymultiply_nobias(x_tile,self.weight)
#         # conv_out = torch.mm(x_tile, masked_weight)
#         conv_out = torch.mm(masked_weight, x_tile)
#         #print("conv_out shape:",conv_out.shape)
#         conv_out = conv_out.reshape(N,-1,self.out_channels).permute(0,2,1).reshape(N,self.out_channels,out_h,-1) #[batch_size, output_channel(kernel_size[0]), out_h, out_w]
#         #print("after reshape conv_out shape:",conv_out.shape)
#         return conv_out



# conv2d = torch.nn.Conv2d(3,4,3)
# conv2dtest = MaskedConv2d(3,4,(3,3))
# conv2d2015test = MaskedConv2d2015(3,4,(3,3))
# conv2d2015test.weight = conv2dtest.weight
# conv2d.weight = conv2dtest.weight
# x = torch.randn([5,3,4,4])
# out = conv2dtest(x)
# out2015 = conv2d2015test(x)
# outnn = conv2d(x)
# print(out.shape)
# # print(outnn.equal(out))
# # print(outnn.equal(out2015))
# print("out 2015:")
# print(out2015)
# print("out:")
# print(out)