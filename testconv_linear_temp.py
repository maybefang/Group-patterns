from models import *
import pickle
import torch

def save(model, maskname):
    mask_val = []
    for l in model.modules():
        if isinstance(l, MaskedConv2d) or isinstance(l, MaskedMLP):
            mask_val.append(l.mask)
    with open(maskname,"wb") as file:
        pickle.dump(mask_val,file)
    
    for k in mask_val:
        print(k.shape)

net = masked_vgg(dataset='cifar10', depth=16)
model_dir = 'checkpoint/vgg16_5e-3alpha_lr02_bn64/best_keepratio_model.pth'
net.load_state_dict(
            torch.load(model_dir, map_location=lambda storage, loc: storage), strict=False)

device = torch.device("cuda")
torch.cuda.set_device(1)

net.to(device)

# with open(model_dir,'rb') as file:
#     mask_kv = pickle.load(file)
# mask_val = []
# for k in mask_kv.keys():
#     mask_val.append(mask_kv[k])
#     print(mask_kv[k].shape)
save(net,'mask_vgg16_25.pkl')

mask_dir = 'mask_vgg16_25.pkl'
with open(mask_dir,'rb') as file:
    mask_val = pickle.load(file)
print(mask_val[0])
for i in mask_val:
    print(float(torch.sum(i))/float(i.numel()))

# x = net.conv1(indata)
# x = net.bn1(x)
# x = net.act(x)

# y = dnet.conv1(indata)
# y = dnet.bn1(y)
# y = dnet.act(y)


# x = net.conv2(x)
# x = net.bn2(x)
# x = net.act(x)

# y = dnet.conv2(y)
# y = dnet.bn2(y)
# y = dnet.act(y)


# x = net.pool(x)
# y = dnet.pool(y)

        
# x = net.conv3(x)
# x = net.bn3(x)
# x = net.act(x)

# y = dnet.conv3(y)
# y = dnet.bn3(y)
# y = dnet.act(y)



# x = net.conv4(x)
# x = net.bn4(x)
# x = net.act(x)

# y = dnet.conv4(y)
# y = dnet.bn4(y)
# y = dnet.act(y)



# x = net.pool(x)
# y = dnet.pool(y)



# x = net.conv5(x)
# x = net.bn5(x)
# x = net.act(x)

# y = dnet.conv5(y)
# y = dnet.bn5(y)
# y = dnet.act(y)

# x = net.conv6(x)
# x = net.bn6(x)
# x = net.act(x)

# y = dnet.conv6(y)
# y = dnet.bn6(y)
# y = dnet.act(y)



# x = net.conv7(x)
# x = net.bn7(x)
# x = net.act(x)

# y = dnet.conv7(y)
# y = dnet.bn7(y)
# y = dnet.act(y)




# x = net.pool(x)
# y = dnet.pool(y)




# x = net.conv8(x)
# x = net.bn8(x)
# x = net.act(x)

# y = dnet.conv8(y)
# y = dnet.bn8(y)
# y = dnet.act(y)



# x = net.conv9(x)
# x = net.bn9(x)
# x = net.act(x)

# y = dnet.conv9(y)
# y = dnet.bn9(y)
# y = dnet.act(y)



# x = net.conv10(x)
# x = net.bn10(x)
# x = net.act(x)

# y = dnet.conv10(y)
# y = dnet.bn10(y)
# y = dnet.act(y)




# x = net.pool(x)
# y = dnet.pool(y)




# x = net.conv11(x)
# x = net.bn11(x)
# x = net.act(x)

# y = dnet.conv11(y)
# y = dnet.bn11(y)
# y = dnet.act(y)



# x = net.conv12(x)
# x = net.bn12(x)
# x = net.act(x)

# y = dnet.conv12(y)
# y = dnet.bn12(y)
# y = dnet.act(y)



# x = net.conv13(x)
# x = net.bn13(x)
# x = net.act(x)

# y = dnet.conv13(y)
# y = dnet.bn13(y)
# y = dnet.act(y)




# x = nn.AvgPool2d(2)(x)
# x = x.view(x.size(0), -1)
# y = nn.AvgPool2d(2)(y)
# y = y.view(y.size(0), -1)




# x = net.fc1(x)
# x = net.bn1d(x)
# x = net.act(x)

# y = dnet.fc1(y)
# y = dnet.bn1d(y)
# y = dnet.act(y)



# x = net.fc2(x)
# y = dnet.fc2(y)
###################################################33
# y = net.conv1(x)
# y = net.bn1(y)
# y = net.act(y)
# x = net.conv2(y)
# x = net.bn2(x)
# x = net.act(x)

# x = net.pool(x)
        
# y = net.conv3(x)
# y = net.bn3(y)
# y = net.act(y)
# x = net.conv4(y)
# x = net.bn4(x)
# x = net.act(x)

# x = net.pool(x)

# y = net.conv5(x)
# y = net.bn5(y)
# y = net.act(y)
# x = net.conv6(y)
# x = net.bn6(x)
# x = net.act(x)
# y = net.conv7(x)
# y = net.bn7(y)
# y = net.act(y)

# y = net.pool(y)

# x = net.conv8(y)
# x = net.bn8(x)
# x = net.act(x)
# y = net.conv9(x)
# y = net.bn9(y)
# y = net.act(y)
# x = net.conv10(y)
# x = net.bn10(x)
# x = net.act(x)

# x = net.pool(x)

# y = net.conv11(x)
# y = net.bn11(y)
# y = net.act(y)
# x = net.conv12(y)
# x = net.bn12(x)
# x = net.act(x)
# y = net.conv13(x)
# y = net.bn13(y)
# y = net.act(y)

# x = nn.AvgPool2d(2)(y)
# x = x.view(x.size(0), -1)

# y = net.fc1(x)
# y = net.bn1d(y)
# x = net.act(y)
# y = net.fc2(x)