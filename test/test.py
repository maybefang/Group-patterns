from testdetailinference import MaskedConv2d,DenseConv2d,MaskedMLP,DenseMLP
import torch

mconv = MaskedConv2d(512,512,kernel_size=(3,3),padding=1,bias=False)
dconv = DenseConv2d(512,512,kernel_size=(3,3),padding=1,bias=False)
#dconv = torch.nn.Conv2d(128,512,kernel_size=(3,3),bias=False)
x = torch.randn(64,512,28,28).cuda()
lx = torch.randn(64,4096).cuda()
maskseed = torch.randint(0,10,[512,3,3])
mask=(maskseed>8).type('torch.IntTensor')
mcratio = float(mask.sum())/float(mask.numel())
# print("keep ratio:",mcratio)

ml = MaskedMLP(4096,4096)
# # dl = torch.nn.Linear(512,256,bias=False)
dl = DenseMLP(4096,4096)
mlmaskseed = torch.randint(0,10,[4096])
mlmask = (mlmaskseed>2).type('torch.IntTensor')
mlratio = float(mlmask.sum())/float(mlmask.numel())
print("keep ratio:",mlratio)

mconv.mask.data = mask
mconv.reinit_weight()

ml.mask.data = mlmask
ml.reinit_weight()


# print(x.device)

mconv.cuda()
dconv.cuda()

ml.cuda()
dl.cuda()

loops=100
# for i in range(100):
#     mconv(x)
#     # dconv(x)
for i in range(100):
    # ml(lx)
    dl(lx)
m_index_select=0
m_im2col = 0
m_mm = 0
m_reshape = 0
m_pruning = 0
m_in = 0
tm = 0

d_w_im2col = 0
d_im2col = 0
d_mm = 0
d_reshape = 0
d_in = 0
td = 0

m_time = 0
d_time = 0
start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
# start.record()
# for i in range(loops):
#     d_out,d_im2col_time, d_w_im2col_time, d_mm_time, d_reshape_time = dconv(x)
#     # _ = dconv(x)
#     # _ = dl(lx)
#     d_w_im2col += d_w_im2col_time
#     d_im2col += d_im2col_time
#     d_mm += d_mm_time
#     d_reshape += d_reshape_time
#     # td += td_time
# end.record()
# torch.cuda.synchronize()
# td = start.elapsed_time(end)

# start.record()
# for i in range(loops):
#     m_out,m_im2col_time,m_index_select_time,m_mm_time,m_reshape_time = mconv(x)
#     # _ = mconv(x)
#     # _ = ml(lx)
#     m_index_select += m_index_select_time
#     m_im2col += m_im2col_time
#     m_mm += m_mm_time
#     m_reshape += m_reshape_time
#     # tm += tm_time
# end.record()
# torch.cuda.synchronize()
# tm = start.elapsed_time(end)

    # _ = mconv(x)

    
    # _ = dconv(x)

    # start.record()
    # #m=mconv(x)
for i in range(loops):
    _,pruning_time,mm_time=ml(lx)
    # _,m_time=ml(lx)
    # end.record()
    # torch.cuda.synchronize()
    # tm += start.elapsed_time(end)
    # m_in+=m_time
    m_pruning += pruning_time
    m_mm += mm_time
    # tm += m_time

    # start.record()
    # #d=dconv(x)
# for i in range(loops):
#     _, d_mm_time = dl(lx)
#     # _,d_time=dl(lx)
#     # end.record()
#     # torch.cuda.synchronize()
#     # td += start.elapsed_time(end)
#     # d_in += d_time
#     d_mm += d_mm_time
#     # td += d_time


# print("m im2col time:%4f"%(m_im2col/loops))#,"          d im2col time:%4f"%(d_im2col/loops))
# print("m index select time:%4f"%(m_index_select/loops))#,"    d weight im2col time:%4f"%(d_w_im2col/loops))
# print("m mm time:%4f"%(m_mm/loops))#,"              d mm time:%4f"%(d_mm/loops))
# print("m reshape time:%4f"%(m_reshape/loops))#,"         d reshape time:%4f"%(d_reshape/loops))
# print("all time:%4f"%(tm/loops))

print("m pruning time:%4f"%(m_pruning/loops))#,"         d mm time:%4f"%(d_mm/loops))
print("m mm time:%4f"%(m_mm/loops))
# print("d mm time:%4f"%(d_mm/loops))

# print("m time:%4f"%(tm/loops),"                d time:%4f"%(td/loops))
# print("m in time:%4f"%(m_in/loops),"                d in time:%4f"%(d_in/loops))
# print("keep ratio:",(mlmask.sum().type('torch.FloatTensor')/mlmask.numel()))
# print("keep ratio:",(mlmask.sum().type('torch.FloatTensor')/mlmask.numel()))