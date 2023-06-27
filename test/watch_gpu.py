from testdetailinference import MaskedConv2d,DenseConv2d,MaskedMLP,DenseMLP
import torch

mconv = MaskedConv2d(512,512,kernel_size=(3,3),padding=1,bias=False)
#dconv = DenseConv2d(512,512,kernel_size=(3,3),padding=1,bias=False)
#dconv = torch.nn.Conv2d(128,512,kernel_size=(3,3),bias=False)
x = torch.randn(2,512,2,2)
lx = torch.randn(64,512)
maskseed = torch.randint(0,10,[512,3,3])
mask=(maskseed>8).type('torch.IntTensor')
# print(mask.sum(),mask.numel())

# ml = MaskedMLP(512,256)
# # dl = torch.nn.Linear(512,256,bias=False)
# dl = DenseMLP(512,256)
# mlmaskseed = torch.randint(0,10,[512])
# mlmask = (mlmaskseed>8).type('torch.IntTensor')

mconv.mask.data = mask.cuda()
mconv.reinit_weight()

# ml.mask.data = mlmask
# ml.reinit_weight()

mconv.cuda()
# lx.cuda()
x.cuda()

loops=1
for i in range(1):
    mconv(x)
    
for i in range(loops):
    
    _ = mconv(x)
