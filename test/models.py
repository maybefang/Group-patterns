defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class masked_vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(masked_vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.Sequential(
              MaskedMLP(cfg[-1], 512),
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
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        x = self.feature(x)
        end.record()
        torch.cuda.synchronize()
        #print("masked vgg feature:",start.elapsed_time(end))
        ft = start.elapsed_time(end)

        start.record()
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        end.record()
        torch.cuda.synchronize()
        #print("masked vgg pool:",start.elapsed_time(end))
        pt = start.elapsed_time(end)

        start.record()
        y = self.classifier(x)
        end.record()
        torch.cuda.synchronize()
        #print("masked linear:",start.elapsed_time(end))
        ct = start.elapsed_time(end)
        return y,ft,pt,ct

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



class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(masked_vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.Sequential(
              nn.Linear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_classes)
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
                conv2d = DenseConv2d(in_channels, v, kernel_size=(3, 3), padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        x = self.feature(x)
        end.record()
        torch.cuda.synchronize()
        #print("masked vgg feature:",start.elapsed_time(end))
        ft = start.elapsed_time(end)

        start.record()
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        end.record()
        torch.cuda.synchronize()
        #print("masked vgg pool:",start.elapsed_time(end))
        pt = start.elapsed_time(end)

        start.record()
        y = self.classifier(x)
        end.record()
        torch.cuda.synchronize()
        #print("masked linear:",start.elapsed_time(end))
        ct = start.elapsed_time(end)
        return y,ft,pt,ct

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


class masked_vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(masked_vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = MaskedConv2d(3, self.cfg[0], kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.cfg[0])
        self.conv2 = MaskedConv2d(self.cfg[0], self.cfg[1], kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.cfg[1])

        self.conv3 = MaskedConv2d(self.cfg[1], self.cfg[3], kernel_size=(3, 3), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.cfg[3])
        self.conv4 = MaskedConv2d(self.cfg[3], self.cfg[4], kernel_size=(3, 3), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.cfg[4])

        self.conv5 = MaskedConv2d(self.cfg[4], self.cfg[6], kernel_size=(3, 3), padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.cfg[6])
        self.conv6 = MaskedConv2d(self.cfg[6], self.cfg[7], kernel_size=(3, 3), padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.cfg[7])
        self.conv7 = MaskedConv2d(self.cfg[7], self.cfg[8], kernel_size=(3, 3), padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(self.cfg[8])

        self.conv8 = MaskedConv2d(self.cfg[8], self.cfg[10], kernel_size=(3, 3), padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(self.cfg[10])
        self.conv9 = MaskedConv2d(self.cfg[10], self.cfg[11], kernel_size=(3, 3), padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(self.cfg[11])
        self.conv10 = MaskedConv2d(self.cfg[11], self.cfg[12], kernel_size=(3, 3), padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(self.cfg[12])

        self.conv11 = MaskedConv2d(self.cfg[12], self.cfg[14], kernel_size=(3, 3), padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(self.cfg[14])
        self.conv12 = MaskedConv2d(self.cfg[14], self.cfg[15], kernel_size=(3, 3), padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(self.cfg[15])
        self.conv13 = MaskedConv2d(self.cfg[15], self.cfg[16], kernel_size=(3, 3), padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(self.cfg[16])


        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.Sequential(
              nn.Linear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_classes)
            )
        if init_weights:
            self._initialize_weights()

   

    def forward(self, x):
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        x = self.conv1(x)
        x = self.act(self.bn1(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c1 = start.elapsed_time(end)

        start.record()
        x = self.conv2(x)
        x = self.act(self.bn2(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c2 = start.elapsed_time(end)

        x = self.pool(x)

        start.record()
        x = self.conv3(x)
        x = self.act(self.bn3(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c3 = start.elapsed_time(end)

        start.record()
        x = self.conv4(x)
        x = self.act(self.bn4(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c4 = start.elapsed_time(end)

        x = self.pool(x)

        start.record()
        x = self.conv5(x)
        x = self.act(self.bn5(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c5 = start.elapsed_time(end)

        start.record()
        x = self.conv6(x)
        x = self.act(self.bn6(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c6 = start.elapsed_time(end)

        start.record()
        x = self.conv7(x)
        x = self.act(self.bn7(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c7 = start.elapsed_time(end)

        x = self.pool(x)

        start.record()
        x = self.conv8(x)
        x = self.act(self.bn8(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c8 = start.elapsed_time(end)

        start.record()
        x = self.conv9(x)
        x = self.act(self.bn9(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c9 = start.elapsed_time(end)

        start.record()
        x = self.conv10(x)
        x = self.act(self.bn10(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c10 = start.elapsed_time(end)
        
        x = self.pool(x)
        
        start.record()
        x = self.conv11(x)
        x = self.act(self.bn11(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c11 = start.elapsed_time(end)

        start.record()
        x = self.conv12(x)
        x = self.act(self.bn12(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c12 = start.elapsed_time(end)
        
        start.record()
        x = self.conv13(x)
        x = self.act(self.bn13(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c13 = start.elapsed_time(end)

        start.record()
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        end.record()
        torch.cuda.synchronize()
        #print("dense pool:",start.elapsed_time(end))
        pt = start.elapsed_time(end)

        start.record()
        y = self.classifier(x)
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg classifier:",start.elapsed_time(end))
        ct = start.elapsed_time(end)
        return y,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13

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



class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DenseConv2d(3, self.cfg[0], kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.cfg[0])
        self.conv2 = DenseConv2d(self.cfg[0], self.cfg[1], kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.cfg[1])

        self.conv3 = DenseConv2d(self.cfg[1], self.cfg[3], kernel_size=(3, 3), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.cfg[3])
        self.conv4 = DenseConv2d(self.cfg[3], self.cfg[4], kernel_size=(3, 3), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.cfg[4])

        self.conv5 = DenseConv2d(self.cfg[4], self.cfg[6], kernel_size=(3, 3), padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.cfg[6])
        self.conv6 = DenseConv2d(self.cfg[6], self.cfg[7], kernel_size=(3, 3), padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.cfg[7])
        self.conv7 = DenseConv2d(self.cfg[7], self.cfg[8], kernel_size=(3, 3), padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(self.cfg[8])

        self.conv8 = DenseConv2d(self.cfg[8], self.cfg[10], kernel_size=(3, 3), padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(self.cfg[10])
        self.conv9 = DenseConv2d(self.cfg[10], self.cfg[11], kernel_size=(3, 3), padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(self.cfg[11])
        self.conv10 = DenseConv2d(self.cfg[11], self.cfg[12], kernel_size=(3, 3), padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(self.cfg[12])

        self.conv11 = DenseConv2d(self.cfg[12], self.cfg[14], kernel_size=(3, 3), padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(self.cfg[14])
        self.conv12 = DenseConv2d(self.cfg[14], self.cfg[15], kernel_size=(3, 3), padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(self.cfg[15])
        self.conv13 = DenseConv2d(self.cfg[15], self.cfg[16], kernel_size=(3, 3), padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(self.cfg[16])


        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.Sequential(
              nn.Linear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        x = self.conv1(x)
        x = self.act(self.bn1(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c1 = start.elapsed_time(end)

        start.record()
        x = self.conv2(x)
        x = self.act(self.bn2(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c2 = start.elapsed_time(end)

        x = self.pool(x)

        start.record()
        x = self.conv3(x)
        x = self.act(self.bn3(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c3 = start.elapsed_time(end)

        start.record()
        x = self.conv4(x)
        x = self.act(self.bn4(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c4 = start.elapsed_time(end)

        x = self.pool(x)

        start.record()
        x = self.conv5(x)
        x = self.act(self.bn5(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c5 = start.elapsed_time(end)

        start.record()
        x = self.conv6(x)
        x = self.act(self.bn6(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c6 = start.elapsed_time(end)

        start.record()
        x = self.conv7(x)
        x = self.act(self.bn7(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c7 = start.elapsed_time(end)

        x = self.pool(x)

        start.record()
        x = self.conv8(x)
        x = self.act(self.bn8(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c8 = start.elapsed_time(end)

        start.record()
        x = self.conv9(x)
        x = self.act(self.bn9(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c9 = start.elapsed_time(end)

        start.record()
        x = self.conv10(x)
        x = self.act(self.bn10(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c10 = start.elapsed_time(end)
        
        x = self.pool(x)
        
        start.record()
        x = self.conv11(x)
        x = self.act(self.bn11(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c11 = start.elapsed_time(end)

        start.record()
        x = self.conv12(x)
        x = self.act(self.bn12(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c12 = start.elapsed_time(end)
        
        start.record()
        x = self.conv13(x)
        x = self.act(self.bn13(x))
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg feature:",start.elapsed_time(end))
        c13 = start.elapsed_time(end)

        start.record()
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        end.record()
        torch.cuda.synchronize()
        #print("dense pool:",start.elapsed_time(end))
        pt = start.elapsed_time(end)

        start.record()
        y = self.classifier(x)
        end.record()
        torch.cuda.synchronize()
        #print("dense vgg classifier:",start.elapsed_time(end))
        ct = start.elapsed_time(end)
        return y,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, DenseConv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                #if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.reset_parameters()