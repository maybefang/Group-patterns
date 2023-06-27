import math
import torch
import torch.nn as nn
import gemm

"""
Function for activation binarization
"""
#由于infernec.py中全连接层出现问题，剪枝好像是剪的通道数，此文件作为备份（仍然是错的），以防无法运行

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
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        # self.threshold = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(1))
        self.step = BinaryStep.apply
        self.mask = nn.Parameter(torch.ones([out_size]))
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

    def reinit_weight():


    def forward(self, input):
        abs_weight = torch.abs(self.weight)#[cout,cin]
        # threshold = self.threshold.view(abs_weight.shape[0], -1)
        mean_weight = torch.sum(abs_weight.data, dim=1) / float(abs_weight.shape[1])
        if self.threshold.data < 0:
            self.threshold.data.fill_(0.)
        mean_weight = mean_weight - self.threshold  # [cin]
        mask = self.step(mean_weight)  # [cin]
        ratio = torch.sum(mask) / mask.numel()
        ###########################################################
        # print("mlp keep ratio {:.2f}".format(ratio))
        # print("mlp threshold {:3f}".format(self.threshold[0]))
        ##########################################################
        if ratio <= 0.01:
            with torch.no_grad():
                # std = self.weight.std()
                self.threshold.data.fill_(0)
            abs_weight = torch.abs(self.weight)
            # threshold = self.threshold.view(abs_weight.shape[0], -1)
            # threshold = self.threshold
            mean_weight = torch.sum(abs_weight.data, dim=1) / float(abs_weight.shape[1])
            mean_weight = mean_weight - self.threshold
            mask = self.step(mean_weight)
        #self.mask = mask.repeat(abs_weight.shape[1], 1).t()  # mask=[cin],self.mask=[cin,cout]
        self.mask = nn.Parameter(mask,requires_grad=False)#mask不需要更新
        # masked_weight = self.weight * mask
        masked_weight = torch.einsum('ij,i->ij', self.weight, mask)
        output = torch.nn.functional.linear(input, masked_weight, self.bias)
        return output

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
        self.step = BinaryStep.apply
        self.reset_parameters()

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
        weight_tile = self.weight.reshape(-1,weight_shape[1]*weight_shape[2]*weight_shape[3]).t() #reshape the weight to [inchannel*h*w, outchannel]
        #根据权重提取idx
        #weight_sum = torch.sum(weight_tile,dim=1)#shape:[outchannel]
        #idx = [i.item() for i in weight_sum.nonzero()]
        #根据mask提取idx
        #self.mask = torch.sum(self.mask,dim=0) #若存储的mask为[out_c,in_c,w,h],所以要把它变为[in_c,w,h]
        mask = self.mask.reshape(-1)#将mask转换为一维，不然下面一行代码报错
        idx = [i.item() for i in mask.nonzero()]
        idx = torch.tensor(idx)#.cuda()#将idx放在GPU上否则报错
        #if self.weight.is_cuda():
        #    idx = idx.cuda()
        #print("weight_tile.shape:",weight_tile.shape)#[27,64]
        #print("idx:",idx)
        masked_weight = torch.index_select(weight_tile,0,idx)#重新拼接权重矩阵
        self.weight=nn.Parameter(masked_weight)
        self.idx = nn.Parameter(idx,requires_grad=False)
    
    

    def im2col(input_data, out_h, out_w, input_shape): 
        N, C, H, W = input_shape
        #out_h = (H + 2 * pad - ksize) // stride + 1
        #out_w = (W + 2 * pad - ksize) // stride + 1
        
        img = torch.nn.functional.pad(input_data,(self.padding, self.padding, self.padding, self.padding, 0, 0),mode="constant", value=0)

        strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
        A = torch.as_strided(img, size=(N,C,out_h,out_w,self.kernel_size[0],self.kernel_size[1]), stride=strides)
        col = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)

        return col 


    def forward(self, x):
        #weight_shape = self.weight.shape  # [out,in,w,h]
        ## threshold = self.threshold.view(weight_shape[0], -1)
        #weight = torch.abs(self.weight)
        ## weight = weight.view(weight_shape[0], -1)
        #weight = torch.sum(weight.data, dim=0) / float(weight_shape[0])  # weight.shape=[cin,w,h]
        #if self.threshold.data < 0:  # make t>=0
        #    self.threshold.data.fill_(0.)
        #weight = weight - self.threshold
        #mask = self.step(weight)  # mask.shape=[cin,w,h]
        ## mask = mask.view(weight_shape)
        #ratio = torch.sum(mask) / mask.numel()
        #####################################################################
        ## print("conv threshold {:3f}".format(self.threshold[0]))
        ## print("conv keep ratio {:.2f}".format(ratio))
        #######################################################################
        #if ratio <= 0.01:
        #    with torch.no_grad():
        #        self.threshold.data.fill_(0.)
        #    # threshold = self.threshold.view(weight_shape[0], -1)
        #    weight = torch.abs(self.weight)
        #    # weight = weight.view(weight_shape[0], -1)
        #    weight = torch.sum(weight.data, dim=0) / float(weight_shape[0])
        #    weight = weight - self.threshold
        #    mask = self.step(weight)
        #    # mask = mask.view(weight_shape)
        ## self.mask = mask    #save the [cin,w,h] size mask
        ##self.mask = mask.repeat(weight_shape[0], 1, 1, 1)  # old:[cin,w,h] new:[cout,cin,w,h]
        ##masked_weight = torch.einsum('ijkl,jkl->ijkl', self.weight, mask)
        #weight_tile = self.weight.reshape(-1,weight_shape[1]*weight_shape[2]*weight_shape[3]).t() #reshape the weight to [inchannel*h*w, outchannel]
        #mask = mask.reshape(-1)#将mask转换为一维，不然下面一行代码报错
        #idx = [i.item() for i in mask.nonzero()]
        #idx = torch.tensor(idx).cuda()#将idx放在GPU上否则报错
        #masked_weight = torch.index_select(weight_tile,0,idx)# spliced into a new matrix
        #由此以上为用t计算mask并将权重矩阵平铺和重新拼接
        #conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
        #                                      padding=self.padding, dilation=self.dilation, groups=self.groups)

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

        strides = (C*H*W, H*W, W*self.stride, self.stride, W*self.stride, self.stride)
        A = torch.as_strided(img, size=(N,C,out_h,out_w,k_h,k_w), stride=strides)
        #x_tile = A.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w,-1)
        x_tile = A.permute(1, 4, 5, 0, 2, 3).reshape(-1, N*out_h*out_w)
        ############################################################################################

        #x_tile = self.im2col(x,out_h,out_w,x_shape)
        #print("x_tile shape:",x_tile.shape)
        x_tile = torch.index_select(x_tile,0,self.idx).t()  #输入重新拼接,因为im2col中已经按照转置排序，此处进行行跳过,然后转置以进行之后计算
        #conv_out = gemm.mymultiply_nobias(x_tile,masked_weight)
        #conv_out = gemm.mymultiply_nobias(x_tile,self.weight)
        conv_out = torch.mm(x_tile,self.weight)
        #print("conv_out shape:",conv_out.shape)
        conv_out = conv_out.reshape(N,-1,self.out_channels).permute(0, 2, 1).reshape(N,self.out_channels,out_h,-1) #[batch_size, output_channel(kernel_size[0]), out_h, out_w]
        #print("after reshape conv_out shape:",conv_out.shape)
        return conv_out
