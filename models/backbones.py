# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import CSIResNet
from .sdaresnet import ResNet,BasicBlock


def Resnet_enc(input_shape, hparams, freeze=None):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 2:  # [B,C,T]
        return CSIResNet(input_shape)
    elif len(input_shape) ==3 and input_shape[1]==224:
        return ResNet(BasicBlock,[2,2,2,2])
    elif len(input_shape) == 3: # [B,A,C,T]
        return None # TODO 返回VIT
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")


def Enc_init(backbonechoice, args):
    """Auto-select an appropriate featurizer for the given input shape."""
    if backbonechoice == 'CSIResNet':  # [B,C,T]
        model = CSIResNet(args.inputshape)
    elif backbonechoice == 'ResNet':  # [B,A,C,T]
        model = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            data_inchannel=args.inputshape[0],
        )
    elif backbonechoice == 'ViT':  # [B,A,C,T]
        inchannels = 3 if args.fda_arc_choice == 'fda' else 1
        model = MAEViT_CSI(
            arch=args.vit_arch,
            mask_ratio=args.vit_mask_ratio,
            embed_dims=args.vit_feat_dim,
            num_layers=args.vit_num_layers,
            num_heads=args.vit_att_num_heads,
            feedforward_channels=args.vit_feedforward_channels,
            input_size=args.inputshape,
            patch_size=args.patch_size,
            in_channels=inchannels,
        )
    else:
        raise ValueError(f"Unknown backbone choice: {backbonechoice}")

    # 冻结部分层，仅保留最后 k 层训练
    if hasattr(args, 'unfreeze_last_k') and args.unfreeze_last_k >= 0:
        model.freeze_backbone_except_last_k(k=args.unfreeze_last_k)

    return model



class AlexFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(AlexFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=(11, 5), stride=2, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.pad2 = nn.ZeroPad2d(2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.pad1 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(4, 3), stride=1, padding=0)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.l4 = nn.AdaptiveAvgPool2d(1)

        # 假设 Flatten 后维度为固定值，比如 2560
        # self.fc1 = nn.Linear(256, 256)
        # self.fc2 = nn.Linear(256, 1280)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.pad1(x)
        x = self.conv3(x)
        x = self.pad1(x)
        x = self.conv4(x)
        x = self.pad1(x)
        x = self.conv5(x)

        x = self.pool3(x)
        x = self.dropout(x)

        # x = x.view(x.size(0), -1)  # Flatten
        x=self.l4(x).squeeze()
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x



