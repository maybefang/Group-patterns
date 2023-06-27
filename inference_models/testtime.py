import torch

from inference_2015 import MaskedConv2d2015
from inference import MaskedConv2d

# weight_tile = torch.randn(4096,4096).cuda()
x = torch.randn(64,512,28,28).cuda()
conv2d = MaskedConv2d(512,512,(3,3))
conv2015 = MaskedConv2d2015(512,512,(3,3))
maskseed = torch.randint(0,10,[512,3,3])
mask = torch.nn.Parameter((maskseed>8).type('torch.IntTensor'),requires_grad=False)
# mask = torch.tensor([1,0,0,1,1])#将mask转换为一维，不然下面一行代码报错
# boolmask = mask.type('torch.BoolTensor').cuda()
# idx = [i.item() for i in mask.nonzero()]
# idx = torch.tensor(idx).cuda()#将idx放在GPU上否则报错

keepratio = float(torch.sum(mask))/float(mask.numel())
print("keep ratio:",keepratio)

conv2d.mask = mask
conv2015.mask = mask

conv2d.reinit_weight()
conv2015.reinit_weight()

conv2d.cuda()
conv2015.cuda()

rep = 100
start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)


start.record()
for i in range(rep):
    # masked_weight = weight_tile[:,boolmask]#torch.index_select(weight_tile,1,idx)
    out = conv2015(x)
end.record()
torch.cuda.synchronize()
time = start.elapsed_time(end)
print("2015 time:",time)

start.record()
for i in range(rep):
    # masked_weight = weight_tile[boolmask,:]#torch.index_select(weight_tile,0,idx)
    out = conv2d(x)
end.record()
torch.cuda.synchronize()
time = start.elapsed_time(end)
print("our time:",time)


start.record()
for i in range(rep):
    # masked_weight = weight_tile[:,boolmask]#torch.index_select(weight_tile,1,idx)
    out = conv2015(x)
end.record()
torch.cuda.synchronize()
time = start.elapsed_time(end)
print("2015 time:",time/rep)

start.record()
for i in range(rep):
    # masked_weight = weight_tile[boolmask,:]#torch.index_select(weight_tile,0,idx)
    out = conv2d(x)
end.record()
torch.cuda.synchronize()
time = start.elapsed_time(end)
print("our time:",time/rep)

