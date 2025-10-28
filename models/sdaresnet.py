import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channel, out_channel, stride=1, padding=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    50层以下的，用BasicBlock
    结构是2个3x3的卷积单元 1个block中channel数不变
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 这里传入stride是因为从第二个stage开始，第一个block中的第一层卷积stride是2 map大小发生了改变
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = norm_layer(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = norm_layer(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identify = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identify = self.downsample(x)
        
        out += identify
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    """
    50层以上的resnet使用BottleNeck
    结构是 三个卷积单元：1x1  3x3  1x1
    """
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv1x1(in_channel, out_channel)
        self.bn1 = norm_layer(out_channel)
        
        # 这里传入stride，是因为从第二个stage开始，第一个block中的第2层卷积stride是2，map大小发生了改变
        self.conv2 = conv3x3(out_channel, out_channel, stride) 
        self.bn2 = norm_layer(out_channel)

        self.conv3 = conv1x1(out_channel, out_channel*self.expansion)
        self.bn3 = norm_layer(out_channel*self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identify = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identify = self.downsample(x)
        
        out += identify
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    
    def __init__(self, block, layers, norm_layer=None,data_inchannel=3):
        super(ResNet, self).__init__()
        if norm_layer is None: 
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.in_channel = 64
        # [N, 3, 224, 224] -> [N, 64, 56, 56]
        self.conv1 = nn.Conv2d(data_inchannel, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #-----------------重点部分 4个stage----------------------------
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # ------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.fc = nn.Linear(512*block.expansion, num_class)

    def _make_layer(self, block, out_channel, blocks, stride=1):
        """
        实现stage，stage由block堆叠而成
        """
        norm_layer = self.norm_layer
        downsample = None

        if stride != 1 or self.in_channel != out_channel*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, out_channel*block.expansion, stride),
                norm_layer(out_channel*block.expansion)
            )
        
        layers = []
        # stage中的第一个block
        layers.append(block(self.in_channel, out_channel, stride, downsample, norm_layer))

        self.in_channel = out_channel*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, out_channel, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def freeze_backbone_except_last_k(self, k=1):
        """
        冻结除了最后 k 个 resnet block和self.classifier之外的所有参数。
        例如 k=1 只训练 layer4，k=2 训练 layer3 和 layer4。
        """
        all_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        layers_to_train = all_layers[-k:]  # 最后 k 个 layer
        layers_to_freeze = all_layers[:-k]  # 其余需要冻结的

        # 冻结前面的 layers
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.maxpool.parameters():
            param.requires_grad = False
    
    def forward_all(self, x):
        """
        DARC专用的前向传播函数
        根据参数返回指定层的输出
        Returns:
            dict: 包含请求层输出的字典
        """
        outputs = {}
        # 第一层操作
        x = self.conv1(x)    # torch.Size([8, 128, 1250])
        x = self.bn1(x)      # torch.Size([8, 128, 1250])
        x = self.relu(x)     # torch.Size([8, 128, 1250])
        x = self.maxpool(x)  # torch.Size([8, 128, 625])
        # 保存第一层输出
        outputs['first'] = x
        # Layer x
        x = self.layer1(x)   # torch.Size([8, 128, 625])
        outputs['layer1'] = x
        x = self.layer2(x)   # torch.Size([8, 128, 313])
        outputs['layer2'] = x
        x = self.layer3(x)   # torch.Size([8, 256, 157])
        outputs['layer3'] = x
        x = self.layer4(x)   # torch.Size([8, 512, 79])
        outputs['layer4'] = x
        # 全局平均池化和展平
        x = self.avgpool(x)  # [N, 512, 1, 1] 或 [N, 2048, 1, 1]
        x = torch.flatten(x, 1)  # [N, 512] 或 [N, 2048]
        # 最终输出
        outputs['final'] = x
        return outputs
    
    def forward_darc(self, x):
        outputs = self.forward_all(x)
        return outputs['final'], outputs['first'].detach()

