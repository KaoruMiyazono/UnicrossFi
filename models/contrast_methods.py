import torch
import torch.nn as nn
import torch.nn.functional as F
from util.optimizer import get_optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np 
import math
import pandas as pd
from .sdaresnet import ResNet, BasicBlock
import torchvision.models as models
from .resnet_wiopen import Res18Featured,LinearAverage,NCACrossEntropy
import time


class ERM(nn.Module):
    """
    Empirical Risk Minimization (ERM)
    相比于Source Only模型，没有任何我们提出的trick，用于无目标域参与的训练场景。
    通常结构包括：特征提取器 + 分类器
    """

    def __init__(self, config,backbone,T_max):

        super(ERM, self).__init__()
        
        self.config=config
        if config.csidataset == 'Widar3.0':
            data_inchan = 2 
        else:
            data_inchan = 3
        # self.feature_extractor = 
        if self.config.backbone=='ResNet':
            self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],data_inchannel=data_inchan)
        else:
            self.feature_extractor = backbone
        
        if config.classify=='linear':  
            self.classifier=nn.Linear(config.last_dim,config.num_classes)
        elif config.classify=='nonlinear':
            self.classifier = nn.Sequential(
                nn.Linear(config.last_dim, config.last_dim *2 ),
                nn.ReLU(),
                nn.Linear(config.last_dim * 2, config.last_dim // 2),
                nn.ReLU(),
                nn.Linear(config.last_dim // 2, config.num_classes)
            )
        else:
             raise ValueError(f"Undefined classifier type: '{config.classify}'. Please choose from ['linear', 'nonlinear'].")
        if self.config.checkpoint_path is not None:
            self.load_pretrained_weights()

        if getattr(self.config, "freeze_enc", False): 
            print("🔒 冻结 feature_extractor 参数，不参与训练")
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            self.optimizer_f = None
            self.scheduler_f = None
        else:
            self.optimizer_f = get_optimizer(self.feature_extractor.parameters(), config)
            self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)

        self.optimizer_c=get_optimizer(self.classifier.parameters(),config)
        self.scheduler_c = CosineAnnealingLR(self.optimizer_c, T_max=T_max, eta_min=0)
        self.criterion = nn.CrossEntropyLoss()
    def load_pretrained_weights(self):
        """
        类内置函数：加载权重
        从 self.config.checkpoint_path 获取路径
        从 self.config.device 获取设备
        """
        weight_path = self.config.checkpoint_path
        device = self.config.device

        print(f"🔹 正在加载权重: {weight_path} 到设备: {device}")

        # 加载权重
        state_dict = torch.load(weight_path, map_location=device)

        # 处理 state_dict（去掉多余前缀）
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            if new_key.startswith('encoder.') or new_key.startswith('feature_extractor.'):
                new_key = new_key.split('.', 1)[1]
            new_state_dict[new_key] = v

        missing_keys, unexpected_keys = self.feature_extractor.load_state_dict(new_state_dict, strict=False)

        print("权重加载完成")
        if missing_keys:
            print(f"缺失参数: {missing_keys}")
        if unexpected_keys:
            print(f"多余参数: {unexpected_keys}")
    def update(self,src_x,y,domain_label,idx,epoch,loader):
        start_time = time.time() 

        if self.optimizer_f is not None:
            self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()

        preds = self.forward(src_x)
        
        # 检查哪些样本是有标签的
        labeled_mask = (y >= 0)
        # 提取有标签和无标签数据
        labeled_preds = preds[labeled_mask]
        labeled_y = y[labeled_mask]
        unlabeled_preds = preds[~labeled_mask]
        # for labeled data
        if labeled_preds.numel() > 0:
            loss = self.criterion(labeled_preds, labeled_y)
        else:
            return preds,torch.tensor(0.0),self.optimizer_f.param_groups[0]['lr'],self.optimizer_f.param_groups[0]['lr']

        #loss = self.criterion(preds, y)
        loss.backward()

        if self.optimizer_f is not None:
            self.optimizer_f.step()
            self.scheduler_f.step()
        self.optimizer_c.step()
        self.scheduler_c.step()
    
        if self.optimizer_f is not None:
            feature_extractor_lr = self.optimizer_f.param_groups[0]['lr']
        else:
            feature_extractor_lr=None
        classifier_lr = self.optimizer_c.param_groups[0]['lr']

        return preds,loss.sum(),feature_extractor_lr,classifier_lr
        
    def forward(self, x):
        x=self.decompose(x)
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out

    def predict(self,x,y):
        out=self.forward(x)
        negative_mask = y < 0
        new_y = y.clone()
        new_y[negative_mask] = torch.abs(new_y[negative_mask] + 1)
        loss = self.criterion(out, new_y)
        return out,loss.sum()
    def predict4time(self,x,y):
        # start_time = time.time()
        out=self.forward(x)
        return out
    
    def get_emb(self,x,y):
        x=self.decompose(x)
        features = self.feature_extractor(x)
        return features
    
    def decompose(self,x):
        if self.config.csidataset=='Widar3.0':
            b,a,v,T,c=x.shape
            x=x.permute(0,1,2,4,3)
            x=x.reshape(b,a,c*v,T)
        elif self.config.csidataset=='SignFi':
            b,a,c,v,T=x.shape
            x=x.permute(0,1,3,2,4)
            if self.config.backbone=='CSIResNet':
                b,a,c,v,t=x.shape
                x=x.reshape(b,a,c*v,T)
        else:
            b,a,c,v,T=x.shape
            x=x.reshape(b,a,c*v,T)
        return x

    
# ###################################################IJCAI 2021 AdvSKM##################################################
class MMD_loss(nn.Module): 
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss
class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, feaure_Len,hidden_dim):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(feaure_Len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
class Cosine_act(nn.Module):
    def __init__(self):
        super(Cosine_act, self).__init__()

    def forward(self, input):
        return torch.cos(input)
cos_act = Cosine_act()
class AdvSKM_Disc(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self,configs):
        """Init discriminator."""
        super(AdvSKM_Disc, self).__init__()

        self.input_dim = configs.features_len * configs.final_out_channels
        self.hid_dim = 128
        configs.disc_hid_dim=64
        self.branch_1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            cos_act,
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.Linear(self.hid_dim // 2, self.hid_dim // 2),
            nn.BatchNorm1d(self.hid_dim // 2),
            cos_act
        )
        self.branch_2 = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.BatchNorm1d(configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim // 2),
            nn.Linear(configs.disc_hid_dim // 2, configs.disc_hid_dim // 2),
            nn.BatchNorm1d(configs.disc_hid_dim // 2),
            nn.ReLU())

    def forward(self, input):
        """Forward the discriminator."""
        out_cos = self.branch_1(input)
        out_rel = self.branch_2(input)
        total_out = torch.cat((out_cos, out_rel), dim=1)
        return total_out
class AdvSKM(nn.Module):
    def __init__(self, config,backbone,T_max):
        super(AdvSKM,self).__init__()
        
        # gesture network
        self.config=config
        if self.config.backbone=='ResNet':
            self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],data_inchannel=2)
        else:
            self.feature_extractor = backbone
        self.classifier_g = nn.Linear(512,config.num_classes)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mmd_loss = MMD_loss()

        import argparse
        configs_1 = argparse.Namespace()
        configs_1.features_len=1 
        configs_1.final_out_channels=512
        configs_1.DSKN_disc_hid=64

        self.AdvSKM_embedder = AdvSKM_Disc(configs_1)

        self.optimizer_f = get_optimizer(self.feature_extractor.parameters(),config)
        self.optimizer_g = get_optimizer(self.classifier_g.parameters(),config)
        self.optimizer_disc=get_optimizer(self.AdvSKM_embedder.parameters(),config)
        self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)
        self.scheduler_c = CosineAnnealingLR(self.optimizer_g, T_max=T_max, eta_min=0)
        self.scheduler_disc = CosineAnnealingLR(self.optimizer_disc, T_max=T_max, eta_min=0)
        self.criterion = nn.CrossEntropyLoss()


    def update(self,src_x,trg_x,y,epoch):
        # pass
        src_x_de,trg_x_de=self.decompose(src_x),self.decompose(trg_x)
        
        if self.config.csidataset=='CSIDA':
            src_feat=self.feature_extractor(src_x_de[:,0,:,:])+self.feature_extractor(src_x_de[:,1,:,:])+self.feature_extractor(src_x_de[:,2,:,:])
            trg_feat=self.feature_extractor(trg_x_de[:,0,:,:])+self.feature_extractor(trg_x_de[:,1,:,:])+self.feature_extractor(trg_x_de[:,2,:,:])
        else:

            src_feat=self.feature_extractor(src_x_de[:,0,:,:])+self.feature_extractor(src_x_de[:,1,:,:])
            trg_feat=self.feature_extractor(trg_x_de[:,0,:,:])+self.feature_extractor(trg_x_de[:,1,:,:])
        src_pred=self.classifier_g(src_feat)
        source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
        target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
        mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
        mmd_loss.requires_grad = True
        self.optimizer_disc.zero_grad()
        mmd_loss.backward()
        self.optimizer_disc.step()
        self.scheduler_disc.step()

        src_cls_loss = self.criterion(src_pred, y)

        # domain loss.
        source_embedding_disc = self.AdvSKM_embedder(src_feat)
        target_embedding_disc = self.AdvSKM_embedder(trg_feat)

        mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
        mmd_loss_adv.requires_grad = True

        # calculate the total loss
        loss =  self.config.domain_loss_weight * mmd_loss_adv + self.config.src_loss_weight * src_cls_loss

        # update optimizer
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()

        loss.backward()
        self.optimizer_f.step()
        self.optimizer_g.step()
        
        self.scheduler_f.step()
        self.scheduler_c.step()
        
        loss=loss
        
        return src_pred,loss.sum(),self.optimizer_f.param_groups[0]['lr'],self.optimizer_f.param_groups[0]['lr']
        
    def predict(self, x,y):
        x_de=self.decompose(x)
        
        if self.config.csidataset=='CSIDA':
            pred=self.classifier_g(self.feature_extractor(x_de[:,0,:,:])+self.feature_extractor(x_de[:,1,:,:])+self.feature_extractor(x_de[:,2,:,:]))
        else:
            pred=self.classifier_g(self.feature_extractor(x_de[:,0,:,:])+self.feature_extractor(x_de[:,1,:,:]))

        loss = self.criterion(pred, y)
        return pred,loss.sum()
    
    def decompose(self,x):
        if self.config.csidataset=='Widar3.0':
            b,a,v,T,c=x.shape
            x=x.permute(0,1,2,4,3)
        
        if self.config.backbone=='ResNet':
            pass
        else: 
            b,a,c,v,T=x.shape
            x=x.reshape(b,a,c*v,T)
        return x
    
# ###################################################IJCAI 2021 AdaSKM##################################################

#####################################################CoTMix TAI2024#####################################################
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        self.batch_size=zis.shape[0]
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    
class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)    
class SupConLoss_cotmix(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, device, temperature=0.2, contrast_mode='all'):
        super(SupConLoss_cotmix, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = self.device  # 'cuda' #(torch.device('cuda')
        # if features.is_cuda
        # else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        eps=1e-8
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - self.temperature * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# NOTE Having changes compared with origin version
class CoTMix(nn.Module):
    def __init__(self, config,backbone,T_max):
        super(CoTMix,self).__init__()
        
        self.config=config
        if self.config.backbone=='ResNet':
            self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],data_inchannel=2)
        else:
            self.feature_extractor = backbone
        
        if config.classify=='linear': #线性分类头 
            self.classifier=nn.Linear(config.last_dim,config.num_classes)
        elif config.classify=='nonlinear':
            self.classifier = nn.Sequential(
                nn.Linear(config.last_dim, config.last_dim *2 ),
                nn.ReLU(),
                nn.Linear(config.last_dim * 2, config.last_dim // 2),
                nn.ReLU(),
                nn.Linear(config.last_dim // 2, config.num_classes)
            )
        device=config.device
        
  
        self.cross_entropy = nn.CrossEntropyLoss()
        self.contrastive_loss = NTXentLoss(device, config.batch_size, 0.2, True)
        self.entropy_loss = ConditionalEntropyLoss()
        self.sup_contrastive_loss = SupConLoss_cotmix(device)


        self.optimizer_f = get_optimizer(self.feature_extractor.parameters(),config)
        self.optimizer_g = get_optimizer(self.classifier.parameters(),config)
        self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)
        self.scheduler_g = CosineAnnealingLR(self.optimizer_g, T_max=T_max, eta_min=0)
        self.step=0
        
    def decompose(self,x):
        if self.config.csidataset=='Widar3.0':
            b,a,v,T,c=x.shape
            x=x.permute(0,1,2,4,3)
        if self.config.backbone=='ResNet':
            pass
        else: 
            b,a,c,v,T=x.shape
            x=x.reshape(b,a,c*v,T)
        return x


    def update(self, src_x,trg_x,src_y,epoch):

        src_dominant, trg_dominant = self.temporal_mixup(src_x, trg_x)
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        src_x_de,trg_x_de=self.decompose(src_x),self.decompose(trg_x)
        src_dominant_de,trg_dominant_de=self.decompose(src_dominant),self.decompose(trg_dominant)
        
        if self.config.csidataset=='CSIDA':
            src_orig_feat = self.feature_extractor(src_x_de[:,0,:,:])+self.feature_extractor(src_x_de[:,1,:,:])+self.feature_extractor(src_x_de[:,2,:,:])
            trg_orig_feat = self.feature_extractor(trg_x_de[:,0,:,:])+self.feature_extractor(trg_x_de[:,1,:,:])+self.feature_extractor(trg_x_de[:,2,:,:])
        else:
            src_orig_feat = self.feature_extractor(src_x_de[:,0,:,:])+self.feature_extractor(src_x_de[:,1,:,:])
            trg_orig_feat = self.feature_extractor(src_x_de[:,0,:,:])+self.feature_extractor(src_x_de[:,1,:,:])
            
            
        src_orig_logits = self.classifier(src_orig_feat)

        
        trg_orig_logits = self.classifier(trg_orig_feat)

        src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
        loss = src_cls_loss * self.config.src_cls_weight

        trg_entropy_loss = self.entropy_loss(trg_orig_logits)
        loss += trg_entropy_loss * self.config.trg_entropy_weight
        # -----------  Auxiliary losses
        # Extract source-dominant mixup features.
        if self.config.csidataset=='CSIDA':
            src_dominant_feat = self.feature_extractor(src_dominant_de[:,0,:,:])+self.feature_extractor(src_dominant_de[:,1,:,:])+self.feature_extractor(src_dominant_de[:,2,:,:])
        else:
            src_dominant_feat = self.feature_extractor(src_dominant_de[:,0,:,:])+self.feature_extractor(src_dominant_de[:,1,:,:])
        src_dominant_logits = self.classifier(src_dominant_feat)

        # supervised contrastive loss on source domain side
        src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
        src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
        loss += src_supcon_loss * self.config.src_supCon_weight

        # Extract target-dominant mixup features.
        if self.config.csidataset=='CSIDA':
            trg_dominant_feat = self.feature_extractor(trg_dominant_de[:,0,:,:])+self.feature_extractor(trg_dominant_de[:,1,:,:])+self.feature_extractor(trg_dominant_de[:,2,:,:])
        else:
            trg_dominant_feat = self.feature_extractor(trg_dominant_de[:,0,:,:])+self.feature_extractor(trg_dominant_de[:,1,:,:])
        trg_dominant_logits = self.classifier(trg_dominant_feat)

        # Unsupervised contrastive loss on target domain side
        # print(trg_orig_logits.shape,trg_dominant_logits.shape)
        trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
        loss += trg_con_loss * self.config.trg_cont_weight
        # print(f"src_cls_loss: {src_cls_loss.item()}, trg_entropy_loss: {trg_entropy_loss.item()}, src_supcon_loss: {src_supcon_loss.item()}, trg_con_loss: {trg_con_loss.item()}")
        loss.backward()
        self.optimizer_f.step()
        self.optimizer_g.step()
        self.scheduler_f.step()
        self.scheduler_g.step()

        return src_dominant_logits,loss.sum(),self.optimizer_f.param_groups[0]['lr'],self.optimizer_f.param_groups[0]['lr']
        

    def predict(self,x,y):
        x_de=self.decompose(x)
        if self.config.csidataset=='CISDA':
            out=self.classifier(self.feature_extractor(x_de[:,0,:,:])+self.feature_extractor(x_de[:,1,:,:])+self.feature_extractor(x_de[:,2,:,:]))
        else:
            out=self.classifier(self.feature_extractor(x_de[:,0,:,:])+self.feature_extractor(x_de[:,1,:,:]))
        loss = self.cross_entropy(out, y)
        return out,loss.sum() 
    def temporal_mixup(self,src_x, trg_x):
    
        # mix_ratio = round(0.9, 2)
        mix_ratio =self.config.mix_ratio
        temporal_shift = self.config.temporal_shift
        h = temporal_shift // 2  # half

        # src_dominant = mix_ratio * src_x + (1 - mix_ratio) * \
        #             torch.mean(torch.stack([torch.roll(trg_x, -i, 2) for i in range(-h, h)], 2), 2)

        # trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * \
        #             torch.mean(torch.stack([torch.roll(src_x, -i, 2) for i in range(-h, h)], 2), 2)
        # print(src_x.shape)
        # CSIDA要换成4
        if self.config.csidataset=='Widar3.0':
            src_x=src_x.permute(0,1,2,4,3)
            trg_x=trg_x.permute(0,1,2,4,3)


        trg_x_rolled = torch.stack([torch.roll(trg_x, -i, dims=4) for i in range(-h, h)], dim=4)
        trg_x_avg = torch.mean(trg_x_rolled, dim=4)

        # 对源域进行多尺度平移并求平均
        src_x_rolled = torch.stack([torch.roll(src_x, -i, dims=4) for i in range(-h, h)], dim=4)
        src_x_avg = torch.mean(src_x_rolled, dim=4)
        

        src_dominant = mix_ratio * src_x + (1 - mix_ratio) * trg_x_avg
        trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * src_x_avg
        if self.config.csidataset=='Widar3.0':
            return src_dominant.permute(0,1,2,4,3), trg_dominant.permute(0,1,2,4,3)
        else:
            return src_dominant,trg_dominant
    
##############################################CoTMix TAI2024######################################################################################################

##############################################WiSDA TMC 2025 #####################################################################################################
class SDA_LOSS(nn.Module):
    def __init__(self, class_num=6):
        super(SDA_LOSS, self).__init__()
        self.class_num = class_num

    def sda_loss(self, source, target, s_label, t_label, r, r1):
        """
        source: (batch, feat_dim)
        target: (batch, feat_dim)
        s_label: (batch,)
        t_label: (batch,)
        r: (batch,) 权重
        r1: 温度因子
        """

        m, n = source.shape
        _, p = target.shape

        # Normalize
        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

        # 相似度矩阵
        result_horizontal = torch.cat([source, target], dim=0)  # [2m, n]
        sim = torch.matmul(result_horizontal, result_horizontal.T)  # [2m, 2m]
        

        # 构建权重矩阵 rx3 = rx1 x rx2
        x = torch.ones(m, device=source.device)  # 源域权重默认是1
        rx = torch.cat([x, r], dim=0)  # [2m]
        rx1 = rx.unsqueeze(1)  # [2m, 1]
        rx2 = rx.unsqueeze(0)  # [1, 2m]
        rx3 = torch.matmul(rx1, rx2)  # [2m, 2m]
        
        

        # 加权相似度矩阵
        sim = sim * rx3

        # 构建正负样本掩码
        label = torch.cat([s_label.view(-1), t_label.view(-1)], dim=0)  # [2m]
        mask_p = (label.unsqueeze(0) == label.unsqueeze(1)).float()  # 正样本掩码
        mask_n = (label.unsqueeze(0) != label.unsqueeze(1)).float()  # 负样本掩码

        # 排除对角线（自己和自己）
        matrix = torch.ones((2 * m, 2 * m), dtype=torch.bool, device=source.device)
        matrix.fill_diagonal_(False)

        # 计算 soft contrastive loss
        # print(r1)
        # print(torch.max(sim/r1))
        # print(torch.mean(sim/r1))
        sim_scaled = torch.exp(sim / r1)  # [2m, 2m]
        # sim_scaled = torch.exp(sim )  # [2m, 2m]
        
        nominator = sim_scaled * mask_p 
        denominator = sim_scaled * mask_n  

        
        a = (nominator.sum(dim=1) / (mask_p.sum(dim=1) + 1e-8))
        b = denominator.sum(dim=1) + 1e-8
        loss_partial = -torch.log(a / b)
        if torch.isnan(a).any() or torch.isinf(a).any():
            print("a 出现异常！")
            print("nominator:", nominator.sum(dim=1))
            print("mask_p.sum:", mask_p.sum(dim=1))

        if torch.isnan(b).any() or torch.isinf(b).any():
            print("b 出现异常！")
            print("denominator:", denominator.sum(dim=1))
            
        
        if torch.isnan(a).any() or torch.isnan(b).any() or torch.isnan(loss_partial).any():
            print("a", a)
            print("b", b)
            print("a/b", a/b)
            print("log(a/b)", torch.log(a/b))
            print("loss_partial", loss_partial.mean())
            exit(0)

        loss = loss_partial.mean()
        return loss

class WiSDA(nn.Module):
    def __init__(self, config, backbone, T_max):
        super(WiSDA, self).__init__()
        self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2])
        self.loss_function = SDA_LOSS(class_num=config.num_classes)
        self.bottle_neck = False
        self.config=config

        if self.bottle_neck:
            self.bottle = nn.Linear(config.last_dim, 256)
            self.classifier = nn.Linear(256, config.num_classes)
        else:
            self.classifier = nn.Linear(config.last_dim, config.num_classes)

        self.optimizer_f = get_optimizer(self.feature_extractor.parameters(), config)
        self.optimizer_c = get_optimizer(self.classifier.parameters(), config)

        self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)
        self.scheduler_c = CosineAnnealingLR(self.optimizer_c, T_max=T_max, eta_min=0)

        self.criterion = nn.CrossEntropyLoss()
        self.epoch_count = 0  # 记录当前 epoch

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.bottle_neck:
            features = self.bottle(features)
        out = self.classifier(features)
        return out

    def update(self, src_x, trg_x, src_y,epoch):
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()

        src_feat = self.feature_extractor(src_x)
        if self.bottle_neck:
            src_feat = self.bottle(src_feat)
        s_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)
        if self.bottle_neck:
            trg_feat = self.bottle(trg_feat)
        t_pred = self.classifier(trg_feat)

        # 伪标签及其置信度
        pseudo_label = F.softmax(t_pred, dim=-1)
        t_label1 = torch.argmax(pseudo_label, dim=-1)

        max_probs, _ = torch.max(pseudo_label, dim=-1)
        mean = torch.mean(max_probs)
        var = torch.var(max_probs)
        # print(f"var:{var}")
        # var = torch.clamp(torch.var(max_probs), min=1e-3)
        r = torch.exp(-((max_probs - mean) ** 2) / (2 * var + 1e-6))
        r = torch.where(r > mean, torch.ones_like(r), r)

        r1 = math.exp(-(epoch + 5) / 10)
        r1 = torch.tensor(r1, dtype=torch.float32, device=src_x.device)
        r1 = r1.to(dtype=torch.float32, device=src_x.device)
        # r = r.detach()
        # r1 = r1.detach()

        loss_cls = self.criterion(s_pred, src_y)
        loss_da = self.loss_function.sda_loss(src_feat, trg_feat, src_y, t_label1, r, r1)

        loss = loss_cls+loss_da * self.config.wisda_loss_weight
        if not torch.isnan(loss):
            
            loss.backward()
            self.optimizer_f.step()
            self.optimizer_c.step()
            self.scheduler_f.step()
            self.scheduler_c.step()

        self.epoch_count += 1 

        return s_pred, loss.item(), self.optimizer_f.param_groups[0]['lr'], self.optimizer_c.param_groups[0]['lr']

    def predict(self, x, y):
        out = self.forward(x)
        loss = self.criterion(out, y)
        return out, loss.item()
##############################################WiSDA TMC 2025 #####################################################################################################



##############################################Wigrut IEEE THMS##############################################################################################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = torch.nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial1 = BasicConv(3, 3, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.spatial = BasicConv(3, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, bn=True, relu=False)
    def forward(self, x):
        x_out = self.spatial(x)
        scale = F.sigmoid(x_out) 
        return x * scale +x, scale
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(gate_channels, gate_channels // reduction_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale, scale


class DACN(nn.Module):
    def __init__(self, num_classes=10):
        super(DACN, self).__init__()
        self.spa = SpatialGate()
        self.cga = ChannelGate(512)
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512,num_classes)
        
    def forward(self, x):       
        outspa, afspa = self.spa(x)
        out = self.features(outspa) 
        outcga, afcga = self.cga(out)
        out = self.avgpool(outcga)
        out = out.view(out.size(0), -1)
        out_feature=out.clone().detach()
        out = self.fc(out)
        return out,outspa,outcga,afspa,out_feature

class WiGRUNT(nn.Module):
    def __init__(self, config,backbone,T_max):

        super(WiGRUNT, self).__init__()
        self.feature_extractor = backbone
        self.network=DACN(config.num_classes)

         
        self.optimizer_f=get_optimizer(self.network.parameters(),config)

        self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)

        self.criterion = nn.CrossEntropyLoss()
             
    def update(self,src_x,trg_x,y,epoch):

        self.optimizer_f.zero_grad()


        preds = self.forward(src_x)

        loss = self.criterion(preds, y)
        loss.backward()

        self.optimizer_f.step()

        
        self.scheduler_f.step()


        return preds,loss.sum(),self.optimizer_f.param_groups[0]['lr'],self.optimizer_f.param_groups[0]['lr']
    def get_feature(self,x):

        _,_,_,_,out=self.network(x)
        return out

    def forward(self, x):
        
        out,_,_,_,_=self.network(x)
        # out = self.classifier(features)
        return out
    def predict(self,x,y):

        out=self.forward(x)
        loss = self.criterion(out, y)
        return out,loss.sum()
    


##############################################Wigrut IEEE THMS##############################################################################################################



##############################################Wiopen IEEE THMS ##############################################################


class Wiopen(nn.Module):
    def __init__(self, config,backbone,T_max,n_data,train_dataset):

        super(Wiopen, self).__init__()
        # self.feature_extractor = backbone
        self.network=Res18Featured(config.num_classes)
        # TODO 把这个字典引入进来
        self.n_data=n_data
        self.lemniscate = LinearAverage(config.last_dim, self.n_data, config.wiopen_temperature, config.wiopen_memory_momentum)
        self.train_dataset=train_dataset


        self.criterion = NCACrossEntropy(torch.LongTensor(self.train_dataset.source_label_list))


        self.criterion2=nn.MSELoss()

         
        self.optimizer_f=get_optimizer(self.network.parameters(),config)

        self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)

        # self.criterion = nn.CrossEntropyLoss()
             
    def update(self,src_x,trg_x,y,epoch,idx,dfs):

        self.optimizer_f.zero_grad()


        rec,features,outputs1= self.forward(src_x)
        outputs =self.lemniscate(features, idx)  #这里的outputs本质上是一个相似度 （b,N_train_data）


        # print(f'outputs.shape:{outputs.shape},idx: {idx.shape}')
        lossnc, ncmax, ncmean = self.criterion(outputs, features, idx) 
        loss = lossnc+1*self.criterion2(rec,dfs)
        
        loss.backward()

        self.optimizer_f.step()

        
        self.scheduler_f.step()


        return outputs1,loss.sum(),self.optimizer_f.param_groups[0]['lr'],self.optimizer_f.param_groups[0]['lr'],ncmax
        
    def forward(self, x):
        
        rec, features, outputs1 = self.network(x)


        # out = self.classifier(features)
        return rec, features, outputs1
    def predict(self,x,y):

        out=self.forward(x)
        loss = self.criterion(out, y)
        return out,loss.sum()
##############################################Wiopen IEEE THMS ##############################################################



################## Our SupCon GR ARC #########################

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        device = features.device
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # (b,n_views,dim) (b*n_views,dim)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print(anchor_feature.shape)
        # exit(0)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        logits_mask = logits_mask.to(logits.device)
        exp_logits = torch.exp(logits) * logits_mask
        # print(exp_logits.shape)
        # print(logits.shape)
        # print(exp_logits.sum(1, keepdim=True).shape)
        # exit(0)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        # if labels==None:
        #     print(mask_pos_pairs)
        #     print(mask_pos_pairs.shape)
        #     exit(0)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=mask_pos_pairs.dtype, device=mask_pos_pairs.device), mask_pos_pairs)
        mask = mask.to(log_prob.device)
        mask_pos_pairs = mask_pos_pairs.to(log_prob.device)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos 

        loss = loss.view(anchor_count, batch_size).mean() # 
        # print(loss)

        return loss 
    
    def forward_semi(self, features, labels=None, mask=None):

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        device = features.device
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        logits_mask = logits_mask.to(logits.device)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=mask_pos_pairs.dtype, device=mask_pos_pairs.device), mask_pos_pairs)
        mask = mask.to(log_prob.device)
        mask_pos_pairs = mask_pos_pairs.to(log_prob.device)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos 
        loss = loss.view(anchor_count, batch_size).mean(dim=0) # 

        return loss

    def build_sup_mask(self,labels):
        b = labels.shape[0]
        labels = labels.contiguous().view(-1, 1)
        # supervised mask for source (b, b)
        mask = torch.eq(labels, labels.T).float()  
        return mask
    
    def build_unsup_mask(self,batch_size, device=None):
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        return mask

    def build_srctgt_mask(self,batch_size, n_views, device=None):
        # 4. Create boolean masks to identify src and tgt rows/columns
        total_rows = 2 * batch_size * n_views
        block_size = 2 * batch_size
        # Initialize masks 
        src_mask = torch.zeros((total_rows,1), dtype=torch.float32).to(device) # 构建基向量
        tgt_mask = torch.zeros((total_rows,1), dtype=torch.float32).to(device)

        # Populate the masks based on the interleaved structure
        for i in range(n_views):
            # The first `bsz` rows of each `block_size` chunk are from the source
            src_start_index = i * block_size
            src_end_index = src_start_index + batch_size
            src_mask[src_start_index:src_end_index] = 1.0

            # The next `bsz` rows are from the target
            tgt_start_index = src_end_index
            tgt_end_index = tgt_start_index + batch_size
            tgt_mask[tgt_start_index:tgt_end_index] = 1.0
        
        src_mask = torch.matmul(src_mask, src_mask.T)
        tgt_mask = torch.matmul(tgt_mask, tgt_mask.T)
        return src_mask,tgt_mask


    def forward_uda(self,src_features, src_labels, tgt_features, tgt_labels=None, configs=None):
        if len(src_features.shape) < 3 or len(tgt_features.shape) < 3 :
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        # (2*b*views,2*b*views)
        
        if len(src_features.shape) > 3:
            src_features = src_features.view(src_features.shape[0], src_features.shape[1], -1)
        
        if len(tgt_features.shape) > 3:
            tgt_features = tgt_features.view(tgt_features.shape[0], tgt_features.shape[1], -1)
        dvc = src_features.device # device

        # src_features 和 tgt_features 都是 (b, n_views, d)
        features = torch.cat([src_features, tgt_features], dim=0)  # 沿着样本维度拼接 -> (2b, n_views, d)
        if tgt_labels is None: # 如果没有来自tgt的伪标签那就进行赋值
            tgt_labels = torch.arange(configs.num_classes+1, configs.num_classes+1+tgt_features.shape[0], device=dvc, dtype=src_labels.dtype)  # [n+1, n+2, ..., n+bsz]
        labels = torch.cat([src_labels, tgt_labels], dim=0)
        
        all_batchsize = features.shape[0]
        # 构建各个mask
        mask_sup_con = self.build_sup_mask(labels)
        mask_ucon=self.build_unsup_mask(all_batchsize,device=dvc)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 先将features沿着第一个维度(num_views)拆开，然后再将它们沿着第一个维度组合
        contrast_count = features.shape[1] # 就是n_views
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # 计算相似度矩阵后除以温度系数，这个矩阵现在是 [bsz*n,bsz*n]的形状

        # 计算 logits
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        mask_sup_con=mask_sup_con.repeat(anchor_count, contrast_count)
        mask_ucon=mask_ucon.repeat(anchor_count, contrast_count)
        src_mask,tgt_mask = self.build_srctgt_mask(src_features.shape[0],n_views=features.shape[1],device=dvc) # src和tgt在mask中对应的block
        src_mask_supcon = mask_sup_con * src_mask
        tgt_mask_supcon = mask_sup_con * tgt_mask
        src_mask_ucon = mask_ucon * src_mask
        tgt_mask_ucon = mask_ucon * tgt_mask


        logits_mask = torch.ones_like(mask_ucon)
        logits_mask = torch.scatter(
            logits_mask,
            1,
            torch.arange(mask_ucon.shape[0]).view(-1, 1).to(mask_ucon.device),
            0
        )

        # 有监督和无监督针对所有样本的mask，1表示正样本，0表示负样本
        # 每个矩阵的每个维度遵守这样的顺序[view1,view2,...,viewn], 每个view 为[source,target],source 和target中有bsz个样本
        mask_ucon = mask_ucon * logits_mask
        mask_sup_con = mask_sup_con * logits_mask
        src_mask_supcon = src_mask_supcon *logits_mask
        tgt_mask_supcon = tgt_mask_supcon *logits_mask
        src_mask_ucon = src_mask_ucon *logits_mask
        tgt_mask_ucon = tgt_mask_ucon *logits_mask
        
        # compute log_prob
        # 下面先将[bsz*n,bsz*n] 的logit矩阵中每个点的log_softmax计算出来（每个点除以除自己之外的所有点的和）
        logits_mask = logits_mask.to(logits.device)
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算mean项
        sup_mask_pos_pairs = mask_sup_con.sum(1)
        ucon_mask_pos_pairs = mask_ucon.sum(1)
        src_sup_mask_pos_pairs = src_mask_supcon.sum(1)
        tgt_sup_mask_pos_pairs = tgt_mask_supcon.sum(1)
        src_ucon_mask_pos_pairs = src_mask_ucon.sum(1)
        tgt_ucon_mask_pos_pairs = tgt_mask_ucon.sum(1)
        
        # 防止除0 
        sup_mask_pos_pairs = torch.where(sup_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=sup_mask_pos_pairs.dtype, device=sup_mask_pos_pairs.device), sup_mask_pos_pairs)
        ucon_mask_pos_pairs = torch.where( ucon_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=ucon_mask_pos_pairs.dtype, device=ucon_mask_pos_pairs.device), ucon_mask_pos_pairs)
        src_sup_mask_pos_pairs = torch.where( src_sup_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=src_sup_mask_pos_pairs.dtype, device=src_sup_mask_pos_pairs.device), src_sup_mask_pos_pairs)
        tgt_sup_mask_pos_pairs = torch.where( tgt_sup_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=tgt_sup_mask_pos_pairs.dtype, device=tgt_sup_mask_pos_pairs.device), tgt_sup_mask_pos_pairs)
        src_ucon_mask_pos_pairs = torch.where( src_ucon_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=src_ucon_mask_pos_pairs.dtype, device=src_ucon_mask_pos_pairs.device), src_ucon_mask_pos_pairs)
        tgt_ucon_mask_pos_pairs = torch.where( tgt_ucon_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=tgt_ucon_mask_pos_pairs.dtype, device=tgt_ucon_mask_pos_pairs.device), tgt_ucon_mask_pos_pairs)

        # 移植设备
        mask_sup_con=mask_sup_con.to(log_prob.device)
        mask_ucon=mask_ucon.to(log_prob.device)
        sup_mask_pos_pairs=sup_mask_pos_pairs.to(log_prob.device)
        ucon_mask_pos_pairs=ucon_mask_pos_pairs.to(log_prob.device)
        src_sup_mask_pos_pairs=src_sup_mask_pos_pairs.to(log_prob.device)
        tgt_sup_mask_pos_pairs=tgt_sup_mask_pos_pairs.to(log_prob.device)
        src_ucon_mask_pos_pairs=src_ucon_mask_pos_pairs.to(log_prob.device)
        tgt_ucon_mask_pos_pairs=tgt_ucon_mask_pos_pairs.to(log_prob.device)

        # 计算loss
        mean_log_prob_pos_sup=(mask_sup_con * log_prob).sum(1) / sup_mask_pos_pairs
        mean_log_prob_pos_ucon=(mask_ucon * log_prob).sum(1) / ucon_mask_pos_pairs

        mean_log_prob_pos_sup_src=(src_mask_supcon * log_prob).sum(1) / src_sup_mask_pos_pairs
        mean_log_prob_pos_sup_tgt=(tgt_mask_supcon * log_prob).sum(1) / tgt_sup_mask_pos_pairs

        mean_log_prob_pos_ucon_src=(src_mask_ucon * log_prob).sum(1) / src_ucon_mask_pos_pairs
        mean_log_prob_pos_ucon_tgt=(tgt_mask_ucon * log_prob).sum(1) / tgt_ucon_mask_pos_pairs

        # loss
        loss_sup = - (self.temperature / self.base_temperature) * mean_log_prob_pos_sup 
        loss_ucon = - (self.temperature / self.base_temperature) * mean_log_prob_pos_ucon 
        loss_src_sup = - (self.temperature / self.base_temperature) * mean_log_prob_pos_sup_src
        loss_tgt_sup = - (self.temperature / self.base_temperature) * mean_log_prob_pos_sup_tgt
        loss_src_ucon = - (self.temperature / self.base_temperature) * mean_log_prob_pos_ucon_src 
        loss_tgt_ucon = - (self.temperature / self.base_temperature) * mean_log_prob_pos_ucon_tgt 

        
        loss_sup = loss_sup.view(anchor_count, all_batchsize).mean() # 因为我们不用赋值伪标签，所以可以直接整个求解均值
        loss_ucon = loss_ucon.view(anchor_count, all_batchsize).mean() # 
        loss_src_sup = loss_src_sup.view(anchor_count, all_batchsize).mean() #
        loss_tgt_sup = loss_tgt_sup.view(anchor_count, all_batchsize).mean() #
        loss_src_ucon = loss_src_ucon.view(anchor_count, all_batchsize).mean() # 
        loss_tgt_ucon = loss_tgt_ucon.view(anchor_count, all_batchsize).mean() # 

        return loss_sup,loss_ucon,loss_src_ucon,loss_tgt_ucon


    def forward_uda_softlabel(self,src_features, src_labels, tgt_features, tgt_labels, configs=None):
        # 这个函数中将所有的样本标签使用one-hot的形式进行表示,这样方便考虑来自伪标签的软标签
        if len(src_features.shape) < 3 or len(tgt_features.shape) < 3 :
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        # (2*b*views,2*b*views)
        
        if len(src_features.shape) > 3:
            src_features = src_features.view(src_features.shape[0], src_features.shape[1], -1)
        
        if len(tgt_features.shape) > 3:
            tgt_features = tgt_features.view(tgt_features.shape[0], tgt_features.shape[1], -1)
        dvc = src_features.device # device

        # src_features 和 tgt_features 都是 (b, n_views, d)
        features = torch.cat([src_features, tgt_features], dim=0)  # 沿着样本维度拼接 -> (2b, n_views, d)
        #
        source_probs = F.one_hot(src_labels, num_classes=configs.num_classes).float().to(dvc)  # (bsz, num_classes)
        labels = torch.cat([source_probs, tgt_labels], dim=0)
        
        all_batchsize = features.shape[0]
        # 构建各个mask
        mask_sup_con = torch.matmul(labels, labels.T) # [2*bsz, 2*bsz]
        mask_sup_con.fill_diagonal_(1) # 将对角线设置为1
        mask_sup_con[mask_sup_con < configs.pseudo_label_threshold] = 0.0 # 低于0.7的都认为是负样本 即双方都应当大于0.85左右

        mask_ucon=self.build_unsup_mask(all_batchsize,device=dvc)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 先将features沿着第一个维度(num_views)拆开，然后再将它们沿着第一个维度组合
        contrast_count = features.shape[1] # 就是n_views
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # 计算相似度矩阵后除以温度系数，这个矩阵现在是 [bsz*n,bsz*n]的形状

        # 计算 logits
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        mask_sup_con=mask_sup_con.repeat(anchor_count, contrast_count)
        mask_ucon=mask_ucon.repeat(anchor_count, contrast_count)
        src_mask,tgt_mask = self.build_srctgt_mask(src_features.shape[0],n_views=features.shape[1],device=dvc) # src和tgt在mask中对应的block
        src_mask_supcon = mask_sup_con * src_mask
        tgt_mask_supcon = mask_sup_con * tgt_mask
        src_mask_ucon = mask_ucon * src_mask
        tgt_mask_ucon = mask_ucon * tgt_mask


        logits_mask = torch.ones_like(mask_ucon)
        logits_mask = torch.scatter(
            logits_mask,
            1,
            torch.arange(mask_ucon.shape[0]).view(-1, 1).to(mask_ucon.device),
            0
        )

        # 有监督和无监督针对所有样本的mask，1表示正样本，0表示负样本
        # 每个矩阵的每个维度遵守这样的顺序[view1,view2,...,viewn], 每个view 为[source,target],source 和target中有bsz个样本
        mask_ucon = mask_ucon * logits_mask
        mask_sup_con = mask_sup_con * logits_mask
        src_mask_supcon = src_mask_supcon *logits_mask
        tgt_mask_supcon = tgt_mask_supcon *logits_mask
        src_mask_ucon = src_mask_ucon *logits_mask
        tgt_mask_ucon = tgt_mask_ucon *logits_mask
        
        # compute log_prob
        # 下面先将[bsz*n,bsz*n] 的logit矩阵中每个点的log_softmax计算出来（每个点除以除自己之外的所有点的和）
        logits_mask = logits_mask.to(logits.device)
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算mean项
        sup_mask_pos_pairs = mask_sup_con.sum(1)
        ucon_mask_pos_pairs = mask_ucon.sum(1)
        src_sup_mask_pos_pairs = src_mask_supcon.sum(1)
        tgt_sup_mask_pos_pairs = tgt_mask_supcon.sum(1)
        src_ucon_mask_pos_pairs = src_mask_ucon.sum(1)
        tgt_ucon_mask_pos_pairs = tgt_mask_ucon.sum(1)
        
        # 防止除0 
        sup_mask_pos_pairs = torch.where(sup_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=sup_mask_pos_pairs.dtype, device=sup_mask_pos_pairs.device), sup_mask_pos_pairs)
        ucon_mask_pos_pairs = torch.where( ucon_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=ucon_mask_pos_pairs.dtype, device=ucon_mask_pos_pairs.device), ucon_mask_pos_pairs)
        src_sup_mask_pos_pairs = torch.where( src_sup_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=src_sup_mask_pos_pairs.dtype, device=src_sup_mask_pos_pairs.device), src_sup_mask_pos_pairs)
        tgt_sup_mask_pos_pairs = torch.where( tgt_sup_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=tgt_sup_mask_pos_pairs.dtype, device=tgt_sup_mask_pos_pairs.device), tgt_sup_mask_pos_pairs)
        src_ucon_mask_pos_pairs = torch.where( src_ucon_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=src_ucon_mask_pos_pairs.dtype, device=src_ucon_mask_pos_pairs.device), src_ucon_mask_pos_pairs)
        tgt_ucon_mask_pos_pairs = torch.where( tgt_ucon_mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=tgt_ucon_mask_pos_pairs.dtype, device=tgt_ucon_mask_pos_pairs.device), tgt_ucon_mask_pos_pairs)

        # 移植设备
        mask_sup_con=mask_sup_con.to(log_prob.device)
        mask_ucon=mask_ucon.to(log_prob.device)
        sup_mask_pos_pairs=sup_mask_pos_pairs.to(log_prob.device)
        ucon_mask_pos_pairs=ucon_mask_pos_pairs.to(log_prob.device)
        src_sup_mask_pos_pairs=src_sup_mask_pos_pairs.to(log_prob.device)
        tgt_sup_mask_pos_pairs=tgt_sup_mask_pos_pairs.to(log_prob.device)
        src_ucon_mask_pos_pairs=src_ucon_mask_pos_pairs.to(log_prob.device)
        tgt_ucon_mask_pos_pairs=tgt_ucon_mask_pos_pairs.to(log_prob.device)

        # 计算loss
        mean_log_prob_pos_sup=(mask_sup_con * log_prob).sum(1) / sup_mask_pos_pairs
        mean_log_prob_pos_ucon=(mask_ucon * log_prob).sum(1) / ucon_mask_pos_pairs

        mean_log_prob_pos_sup_src=(src_mask_supcon * log_prob).sum(1) / src_sup_mask_pos_pairs
        mean_log_prob_pos_sup_tgt=(tgt_mask_supcon * log_prob).sum(1) / tgt_sup_mask_pos_pairs

        mean_log_prob_pos_ucon_src=(src_mask_ucon * log_prob).sum(1) / src_ucon_mask_pos_pairs
        mean_log_prob_pos_ucon_tgt=(tgt_mask_ucon * log_prob).sum(1) / tgt_ucon_mask_pos_pairs

        # loss
        loss_sup = - (self.temperature / self.base_temperature) * mean_log_prob_pos_sup 
        loss_ucon = - (self.temperature / self.base_temperature) * mean_log_prob_pos_ucon 
        loss_src_sup = - (self.temperature / self.base_temperature) * mean_log_prob_pos_sup_src
        loss_tgt_sup = - (self.temperature / self.base_temperature) * mean_log_prob_pos_sup_tgt
        loss_src_ucon = - (self.temperature / self.base_temperature) * mean_log_prob_pos_ucon_src 
        loss_tgt_ucon = - (self.temperature / self.base_temperature) * mean_log_prob_pos_ucon_tgt 

        
        loss_sup = loss_sup.view(anchor_count, all_batchsize).mean() # 因为我们不用赋值伪标签，所以可以直接整个求解均值
        loss_ucon = loss_ucon.view(anchor_count, all_batchsize).mean() # 
        loss_src_sup = loss_src_sup.view(anchor_count, all_batchsize).mean() #
        loss_tgt_sup = loss_tgt_sup.view(anchor_count, all_batchsize).mean() #
        loss_src_ucon = loss_src_ucon.view(anchor_count, all_batchsize).mean() # 
        loss_tgt_ucon = loss_tgt_ucon.view(anchor_count, all_batchsize).mean() # 

        return loss_sup,loss_ucon,loss_src_sup,loss_tgt_sup,loss_src_ucon,loss_tgt_ucon    


class UniCrossFi_uda_hardpseudo_fbc(nn.Module):
    """
    使用hard pseudo label进行目标域的有监督对比学习
    """
    def __init__(self, config,backbone,T_max):

        super(UniCrossFi_uda_hardpseudo_fbc, self).__init__()
        if config.csidataset == 'Widar3.0':
            data_inchan = 2 
        else:
            data_inchan = 2
        if config.backbone=='ResNet':
            self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],data_inchannel=data_inchan)
        else:
            self.feature_extractor = backbone
        self.config=config
        self.device=torch.device(config.device)

        # neck
        self.neck = nn.Sequential(
                nn.Linear(config.last_dim, config.last_dim ),
                nn.ReLU(),
                nn.Linear(config.last_dim, config.last_dim//4)
            )
        feat_dim = config.last_dim 

        # 手势分类头
        if config.classify=='linear': #线性分类头 
            self.classifier=nn.Linear(feat_dim,config.num_classes)
        elif config.classify=='nonlinear':
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, feat_dim ),
                nn.ReLU(),
                nn.Linear(feat_dim, config.num_classes)
            )
        else:
             raise ValueError(f"Undefined classifier type: '{config.classify}'. Please choose from ['linear', 'nonlinear'].")
         
        self.optimizer_f=get_optimizer(self.feature_extractor.parameters(),config)
        self.optimizer_c=get_optimizer(self.classifier.parameters(),config)
        self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)
        self.scheduler_c = CosineAnnealingLR(self.optimizer_c, T_max=T_max, eta_min=0)
        self.criterion = SupConLoss(temperature=config.temperature)
        self.source_prototype = torch.zeros(config.num_classes, config.last_dim, dtype=torch.float, device=self.device)
        self.target_prototype = torch.zeros(config.num_classes, config.last_dim, dtype=torch.float, device=self.device)

    def h_map(self,x,threshold=0.7, k=10):
        return 1 / (1 + torch.exp(-k * (x - threshold)))
    def randomize(self, x, eps=1e-5):#torch.Size([128, 512])
        sizes = x.size()
        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)  # [b, c, h*w]

        # 原始 mean/std
        orig_mean = x.mean(-1, keepdim=True)   # [b, c, 1] or [b, 1]
        orig_std = x.std(-1, keepdim=True) + eps

        # 打乱得到新的 mean/std
        idx_swap = torch.randperm(sizes[0], device=x.device)
        new_mean = orig_mean[idx_swap]
        new_std  = orig_std[idx_swap]

        alpha = torch.rand(sizes[0], 1, device=x.device)
        if len(sizes) == 4:
            alpha = alpha.unsqueeze(-1)
        new_mean = alpha * orig_mean + (1 - alpha) * new_mean
        new_std  = alpha * orig_std  + (1 - alpha) * new_std

        # 重新参数化
        x_norm = (x - orig_mean) / orig_std
        x_new = x_norm * new_std + new_mean
    
        return x_new.view(*sizes)
    def compute_prototypes(self, train_loader, test_loader=None):
        # 重新初始化 source 和 target prototype
        print("📌 [Info] Start computing prototypes...")
        self.source_prototype = torch.zeros(
            self.config.num_classes, 
            self.config.last_dim, 
            dtype=torch.float, 
            device=self.device
        )
        self.target_prototype = torch.zeros(
            self.config.num_classes, 
            self.config.last_dim, 
            dtype=torch.float, 
            device=self.device
        )

        # 用于统计每个类别累加的样本数
        src_counts = torch.zeros(self.config.num_classes, device=self.device)
        tgt_counts = torch.zeros(self.config.num_classes, device=self.device)

        with torch.no_grad():  # 不计算梯度
            ############## Source 部分 ##############
            for i, (x, y, _, _, _) in enumerate(train_loader):
                x = x.permute(0, 1, 2, 4, 3).to(self.device)  # [bsz, ant, amp_phase, time, sub]
                y = y.to(self.device)
                # print(f'里面{x.shape}')

                feats, logits, embs = self.forward(x)
                features = F.normalize(feats[:, 0, :], dim=-1)  # 归一化

                # 按类别 mask 累加
                for cls in range(self.config.num_classes):
                    mask = (y == cls)
                    if mask.sum() > 0:
                        cls_feats = features[mask]
                        self.source_prototype[cls] += cls_feats.sum(dim=0)
                        src_counts[cls] += mask.sum()

            # 归一化
            for cls in range(self.config.num_classes):
                if src_counts[cls] > 0:
                    self.source_prototype[cls] /= src_counts[cls]

            ############## Target 部分 ##############
            if test_loader is not None:
                for i, (tgt_x, _, _, _, _) in enumerate(test_loader):
                    tgt_x = tgt_x.permute(0, 1, 2, 4, 3).to(self.device)
                    tgt_feats, tgt_logits, _ = self.forward(tgt_x)
                    tgt_feats = F.normalize(tgt_feats[:, 0, :], dim=-1)  # 原样本计算+归一化 仿照FBC

                    # 计算伪标签及置信度
                    probs = F.softmax(tgt_logits[:,0,:], dim=-1)  # [bsz, num_classes]
                    pseudo_y = probs.argmax(dim=-1)  # [bsz]
                    confidence, _ = probs.max(dim=-1)  # [bsz]

                    # 过滤低置信度样本
                    mask_high_conf = confidence >= self.config.pseudo_label_threshold
                    if mask_high_conf.sum() == 0:
                        continue

                    high_conf_feats = tgt_feats[mask_high_conf]
                    high_conf_labels = pseudo_y[mask_high_conf]

                    # 累加高置信度样本
                    for cls in range(self.config.num_classes):
                        mask = (high_conf_labels == cls)
                        if mask.sum() > 0:
                            cls_feats = high_conf_feats[mask]
                            self.target_prototype[cls] += cls_feats.sum(dim=0)
                            tgt_counts[cls] += mask.sum()

                # 归一化
                for cls in range(self.config.num_classes):
                    if tgt_counts[cls] > 0:
                        self.target_prototype[cls] /= tgt_counts[cls]
                    else:
                        self.target_prototype[cls] = self.source_prototype[cls].clone()  # 一个都没有只能用source的了
        print("✅ [Info] Prototypes computed.")
            


    def update(self,src_x,tgt_x,src_y,domain_label,idx,epoch,loader):
        """
        1. epoch<30时，source使用ce+ucon，target使用ucon
        2. epoch>=30时，source使用ce+supcon+ucon，target使用ce伪标签的supcon+ucon
        """
        # learn gesture feature
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        
        src_feats,src_logits,src_emb = self.forward(src_x) # 返回源域输出
        tgt_feats,tgt_logits,tgt_emb = self.forward(tgt_x) #返回目标域输出

        if epoch< self.config.pseudo_start_epoch:
            # 第一阶段，仅仅计算所有数据的con和source的ce
            ce_list = []
            for v in range(src_logits.size(1)):  # 遍历 n_views
                ce_v = F.cross_entropy(src_logits[:, v, :], src_y)
                ce_list.append(ce_v)
            loss_ce = sum(ce_list) / len(ce_list)  # 取平均
            all_emb = torch.cat([src_emb, tgt_emb], dim=0)
            loss_ucon = self.criterion(all_emb)
            loss_con = loss_ucon
            loss_fbc = torch.tensor(0.0, device=self.device)
        
        else:
            # 第二阶段为目标域进行伪标签标注后计算所有对比损失
            # 仅仅计算源域的分类损失
            ce_list = []
            for v in range(src_logits.size(1)):  # 遍历 n_views
                ce_v = F.cross_entropy(src_logits[:, v, :], src_y)
                ce_list.append(ce_v)
            loss_ce = sum(ce_list) / len(ce_list)  # 取平均
            # 伪标签标注
            probs = F.softmax(tgt_logits, dim=-1)  # [num_unlabeled, n_views, num_classes]
            probs_mean = probs.mean(dim=1)  # 每个类的概率 [num_unlabeled, num_classes] 对所有增强样本取均值（和最后的分类步骤统一）
            pseudo_y = probs_mean.argmax(dim=-1)  # [num_unlabeled] 
            confidence, _ = probs_mean.max(dim=-1)  # 伪标签的置信度 [num_unlabeled] 
            #confidence=self.h_map(confidence) # 映射到0-1之间
            tgt_batch_indices = torch.arange(tgt_feats.shape[0], device=pseudo_y.device,dtype=pseudo_y.dtype)
            # 标签值将从 num_classes, num_classes + 1, ... 开始,这确保了它们与真实的类别标签 (0 到 num_classes-1) 不会冲突
            unique_labels_for_low_confidence = self.config.num_classes+ 1 + tgt_batch_indices
            # 使用 torch.where 根据置信度条件进行选择
            tgt_y = torch.where(
                confidence >= self.config.pseudo_label_threshold,      # 条件：置信度是否大于等于0.8
                pseudo_y,                              # 如果是，则采用 argmax 得到的伪标签
                unique_labels_for_low_confidence       # 如果否，则使用 "num_class + 1 + index" 作为唯一标签
            )

            all_emb = torch.cat([src_emb, tgt_emb], dim=0)
            all_y = torch.cat([src_y, tgt_y], dim=0)
            # 将源域和目标域的数据输入进行对比损失计算
            loss_supcon = self.criterion(all_emb, all_y)
            loss_ucon = self.criterion(all_emb)
            loss_con = self.config.beta*loss_supcon  +  (1-self.config.beta)*loss_ucon

            # 开始计算 FBC_loss
            combined_logits = torch.cat([src_logits, tgt_logits], dim=0)  # [bsz_src + bsz_tgt, n_views, num_classes]
            combined_logits = combined_logits[:, 1, :]  # 只取第2个view的logits进行FBC计算
            probs_combined = F.softmax(combined_logits, dim=-1)  # [bsz_src + bsz_tgt, num_classes]
            pseudo_combined = probs_combined.argmax(dim=-1)  # [bsz_src + bsz_tgt]
            confidence_combined, _ = probs_combined.max(dim=-1)  # [bsz_src + bsz_tgt]
            high_conf_mask = confidence_combined >= self.config.pseudo_label_threshold
            if high_conf_mask.sum() > 0:
                high_conf_feats = F.normalize(torch.cat([src_feats, tgt_feats], dim=0)[high_conf_mask][:, 1, :], dim=-1)  # 只取第2个view的feats进行FBC计算
                high_conf_labels = pseudo_combined[high_conf_mask]

                similarity_src = torch.mm(high_conf_feats, self.source_prototype.t())  # [N, classes]
                loss1 = F.cross_entropy(similarity_src, high_conf_labels,reduction="mean")
                similarity_tgr = torch.mm(high_conf_feats, self.target_prototype.t())  # [N, classes]
                loss2 = F.cross_entropy(similarity_tgr, high_conf_labels,reduction="mean")
                loss_fbc = (loss1 + loss2) / 2
            else:
                loss_fbc = torch.tensor(0.0, device=self.device) 


        scale = 1
        loss = loss_ce + scale * loss_con + loss_fbc
        loss.backward() 
        self.optimizer_f.step()
        self.optimizer_c.step()
        self.scheduler_f.step()
        self.scheduler_c.step()
        if epoch<self.config.pseudo_start_epoch:
            loss_dict = {'loss_ce':loss_ce.item(),'loss_con':loss_con.item()}
        else:
            loss_dict = {'loss_ce':loss_ce.item(),
            'loss_con':loss_con.item(),'loss_supcon':loss_supcon.item(),'loss_ucon':loss_ucon.item(),'loss_fbc':loss_fbc.item()}
        for v, ce_v in enumerate(ce_list):
            loss_dict[f'loss_ce_view{v}'] = ce_v.item()
        return src_logits[:,0,:].squeeze(),loss_dict,self.optimizer_f.param_groups[0]['lr'],self.optimizer_c.param_groups[0]['lr']
        
    def forward(self, x):
        x=self.decompose(x)
        feats = self.forward_f(x) # [b, n_views, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)
        # 1. cls接在emb后直接计算
        #logits = self.classifier(embs)
        # 2. cls接在 feature 后计算
        logits = self.classifier(feats)        
        feats = feats.view(b, n_views, -1) # [bsz,nview,d]
        logits = logits.view(b,n_views,-1)
        embs = embs.view(b,n_views,-1)
        return feats,logits,embs

    def forward_f(self, x):
        if self.config.csidataset == 'Widar3.0':
            x_anchor = x[:, 0, :, :, :]                # [b, amp_phase,c, t]
            x_indomain_view = x[:, 1, :, :, :]
            feats_1 = self.feature_extractor(x_anchor) # [b,d]
            feats_2 = self.feature_extractor(x_indomain_view) # [b,d]
            feats_interdomain = self.randomize(feats_2) #[b,d]
            feats = torch.stack([feats_1, feats_2, feats_interdomain], dim=1)  # [b, 3, d]
        else:
            x_1, x_2, x_3 = x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:]
            feats_1 = self.feature_extractor(x_1) # [b,d]
            feats_2 = self.feature_extractor(x_2) # [b,d]
            feats_3 = self.feature_extractor(x_3)
            feats_interdomain2 = self.randomize(feats_2)
            feats_interdomain3 = self.randomize(feats_3)
            feats = torch.stack([feats_1, feats_2, feats_3, feats_interdomain2, feats_interdomain3], dim=1)  # [b, 5,d]
        return feats
        
    def predict(self,x,y):
        # 由于训练的时候是每个view单独求解，为了利用所有的view，在predicting的时候我们使用soft voting的方式
        x=self.decompose(x)
        if self.config.csidataset == 'Widar3.0':
            x_anchor = x[:, 0, :, :, :]                # [b, amp_phase,c, t]
            x_indomain_view = x[:, 1, :, :, :]
            feats_1 = self.feature_extractor(x_anchor) # [b,d]
            feats_2 = self.feature_extractor(x_indomain_view) # [b,d]
            feats = torch.stack([feats_1, feats_2], dim=1)  # [b, 2, d]
        else:
            x_1, x_2, x_3 = x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:]
            feats_1 = self.feature_extractor(x_1) # [b,d]
            feats_2 = self.feature_extractor(x_2) # [b,d]
            feats_3 = self.feature_extractor(x_3)
            feats = torch.stack([feats_1, feats_2, feats_3], dim=1)  # [b, 3, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)
        #logits = self.classifier(embs) # [b*n_views, d]
        logits = self.classifier(feats)    
        #embs = embs.view(b,n_views,-1) 
        logits = logits.view(b,n_views,-1)
        probs = F.softmax(logits, dim=-1)      # [b, n_views, num_classes]
        # step2: 平均概率 (soft-voting)
        probs_mean = probs.mean(dim=1)         # [b, num_classes]
        # step3: log + nll_loss
        log_probs = torch.log(probs_mean + 1e-8)  # 防止log(0)
        loss = F.nll_loss(log_probs, y)
        return probs_mean, loss.sum()

    def decompose(self,x):
        if self.config.csidataset=='Widar3.0': #[bsz,ant,amp+phase,t,c]
            b,a,ap,T,c=x.shape 
            x=x.permute(0,1,2,4,3) # [bsz,ant,amp_phase,sub,time]
            return x
        else:
            # torch.Size([128, 3, 114, 2, 1800]) [bsz,ant,sub,amp_phase,time]
            b,a,c,ap,T=x.shape 
            x=x.permute(0,1,3,2,4)# [bsz,ant,amp_phase,sub,time]
            return x

class UniCrossFi_semidg(nn.Module):
    def __init__(self, config,backbone,T_max):

        super(UniCrossFi_semidg, self).__init__()
        if config.csidataset == 'Widar3.0':
            data_inchan = 2 
        else:
            data_inchan = 2
        if config.backbone=='ResNet':
            self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],data_inchannel=data_inchan)
        else:
            self.feature_extractor = backbone
        self.config=config
        self.device=torch.device(config.device)

        # neck
        self.neck = nn.Sequential(
                nn.Linear(config.last_dim, config.last_dim ),
                nn.ReLU(),
                nn.Linear(config.last_dim, config.last_dim//4)
            )
        feat_dim = config.last_dim
        # 手势分类头
        if config.classify=='linear': #线性分类头 
            self.classifier=nn.Linear(feat_dim,config.num_classes)
        elif config.classify=='nonlinear':
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, feat_dim ),
                nn.ReLU(),
                nn.Linear(feat_dim, config.num_classes)
            )
        else:
             raise ValueError(f"Undefined classifier type: '{config.classify}'. Please choose from ['linear', 'nonlinear'].")
         
        self.optimizer_f=get_optimizer(self.feature_extractor.parameters(),config)
        self.optimizer_c=get_optimizer(self.classifier.parameters(),config)
        self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)
        self.scheduler_c = CosineAnnealingLR(self.optimizer_c, T_max=T_max, eta_min=0)
        self.criterion = SupConLoss(temperature=config.temperature)
    
    def randomize(self, x, eps=1e-5):#torch.Size([128, 512])
        # return x
        ##
        sizes = x.size()
        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)  # [b, c, h*w]

        # 原始 mean/std
        orig_mean = x.mean(-1, keepdim=True)   # [b, c, 1] or [b, 1]
        orig_std = x.std(-1, keepdim=True) + eps

        # 打乱得到新的 mean/std
        idx_swap = torch.randperm(sizes[0], device=x.device)
        new_mean = orig_mean[idx_swap]
        new_std  = orig_std[idx_swap]

        alpha = torch.rand(sizes[0], 1, device=x.device)
        if len(sizes) == 4:
            alpha = alpha.unsqueeze(-1)
        new_mean = alpha * orig_mean + (1 - alpha) * new_mean
        new_std  = alpha * orig_std  + (1 - alpha) * new_std

        # 重新参数化
        x_norm = (x - orig_mean) / orig_std
        x_new = x_norm * new_std + new_mean
    
        return x_new.view(*sizes)

    def update(self,x,y,domain_label,idx,epoch,loader):
        # learn gesture feature
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()

        feats,logits,emb = self.forward(x) #返回三个部分的输出
        # 检查哪些样本是有标签的
        labeled_mask = (y >= 0)
        # 提取有标签和无标签数据
        labeled_feats = feats[labeled_mask]
        labeled_logits = logits[labeled_mask]
        labeled_emb = emb[labeled_mask]
        labeled_y = y[labeled_mask]
        unlabeled_feats = feats[~labeled_mask]
        unlabeled_logits = logits[~labeled_mask]
        unlabeled_emb = emb[~labeled_mask]  # torch.Size([num_sample, num_view, dim])

        if epoch<35:
            # 第一阶段，仅仅计算所有数据的con和有标记数据的ce
            ce_list = []
            for v in range(labeled_logits.size(1)):  # 遍历 n_views
                ce_v = F.cross_entropy(labeled_logits[:, v, :], labeled_y)
                ce_list.append(ce_v)
            loss_ce = sum(ce_list) / len(ce_list)  # 取平均
            loss_ucon = self.criterion(emb)
            loss_con = loss_ucon
            loss_fbc = torch.tensor(0.0, device=loss_ce.device)
        else:
            # 第二阶段为目标域进行伪标签标注后计算所有对比损失
            # 仅仅计算源域的分类损失
            ce_list = []
            for v in range(labeled_logits.size(1)):  # 遍历 n_views
                ce_v = F.cross_entropy(labeled_logits[:, v, :], labeled_y)
                ce_list.append(ce_v)
            loss_ce = sum(ce_list) / len(ce_list)  # 取平均
            # 伪标签标注
            probs = F.softmax(unlabeled_logits, dim=-1)  # [num_unlabeled, n_views, num_classes]
            probs_mean = probs.mean(dim=1)  # 每个类的概率 [num_unlabeled, num_classes] 对所有增强样本取均值（和最后的分类步骤统一）
            pseudo_y = probs_mean.argmax(dim=-1)  # [num_unlabeled] 
            confidence, _ = probs_mean.max(dim=-1)  # 伪标签的置信度 [num_unlabeled] 
            #confidence=self.h_map(confidence) # 映射到0-1之间
            ul_batch_indices = torch.arange(unlabeled_feats.shape[0], device=pseudo_y.device,dtype=pseudo_y.dtype)
            # 标签值将从 num_classes, num_classes + 1, ... 开始,这确保了它们与真实的类别标签 (0 到 num_classes-1) 不会冲突
            unique_labels_for_low_confidence = self.config.num_classes+ 1 + ul_batch_indices
            # 使用 torch.where 根据置信度条件进行选择
            unlabeled_y = torch.where(
                confidence >= self.config.pseudo_label_threshold,      # 条件：置信度是否大于等于0.75
                pseudo_y,                              # 如果是，则采用 argmax 得到的伪标签
                unique_labels_for_low_confidence       # 如果否，则使用 "num_class + 1 + index" 作为唯一标签
            )

            all_emb = torch.cat([labeled_emb, unlabeled_emb], dim=0)
            all_y = torch.cat([labeled_y, unlabeled_y], dim=0)
            # 将源域和目标域的数据输入进行对比损失计算
            loss_supcon = self.criterion(all_emb, all_y)
            loss_ucon = self.criterion(all_emb)
            loss_con = self.config.beta*loss_supcon  +  (1-self.config.beta)*loss_ucon

            # 开始计算 FBC_loss
            combined_logits = torch.cat([labeled_logits, unlabeled_logits], dim=0)  # [bsz_src + bsz_tgt, n_views, num_classes]
            combined_logits = combined_logits[:, 1, :]  # 只取第2个view的logits进行FBC计算
            probs_combined = F.softmax(combined_logits, dim=-1)  
            pseudo_combined = probs_combined.argmax(dim=-1)  
            confidence_combined, _ = probs_combined.max(dim=-1)  
            high_conf_mask = confidence_combined >= self.config.pseudo_label_threshold
            if high_conf_mask.sum() > 0:
                high_conf_feats = F.normalize(torch.cat([labeled_feats, unlabeled_feats], dim=0)[high_conf_mask][:, 1, :], dim=-1)  # 只取第2个view的feats进行FBC计算
                high_conf_labels = pseudo_combined[high_conf_mask]

                similarity_src = torch.mm(high_conf_feats, self.source_prototype.t())  # [N, classes]
                loss1 = F.cross_entropy(similarity_src, high_conf_labels,reduction="mean")
                similarity_tgr = torch.mm(high_conf_feats, self.target_prototype.t())  # [N, classes]
                loss2 = F.cross_entropy(similarity_tgr, high_conf_labels,reduction="mean")
                loss_fbc = (loss1 + loss2) / 2
            else:
                loss_fbc = torch.tensor(0.0, device=self.device) 


        scale = 1
        fbc_scale = 0
        loss = loss_ce + scale * loss_con + fbc_scale*loss_fbc

        loss.backward() 
        self.optimizer_f.step()
        self.optimizer_c.step()
        self.scheduler_f.step()
        self.scheduler_c.step()
        # loss_dict = {'loss_ce':loss_ce.item(),'loss_supcon':loss_supcon.item(),'loss_usupcon':loss_usupcon.item()} #zzy改的
        loss_dict = {'loss_ce':loss_ce.item(),'loss_con':loss_con.item(),'loss_ucon':loss_ucon.item()}

        for v, ce_v in enumerate(ce_list):
            loss_dict[f'loss_ce_view{v}'] = ce_v.item()
        return logits[:,0,:].squeeze(),loss_dict,self.optimizer_f.param_groups[0]['lr'],self.optimizer_c.param_groups[0]['lr']
        
    def forward(self, x):
        x=self.decompose(x)
        feats = self.forward_f(x) # [b, n_views, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)
        # 1. cls接在emb后直接计算
        #logits = self.classifier(embs)
        # 2. cls接在 feature 后计算
        logits = self.classifier(feats)        
        feats = feats.view(b, n_views, -1) # [bsz,nview,d]
        logits = logits.view(b,n_views,-1)
        embs = embs.view(b,n_views,-1)
        return feats,logits,embs
    def get_emb(self,x,y):
        # 由于训练的时候是每个view单独求解，为了利用所有的view，在predicting的时候我们使用soft voting的方式
        x=self.decompose(x)
        feats = self.forward_f(x) # [b, n_views, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)
        # 1. cls接在emb后直接计算
        #logits = self.classifier(embs)
        # 2. cls接在 feature 后计算
        logits = self.classifier(feats)        
        feats = feats.view(b, n_views, -1) # [bsz,nview,d]
        logits = logits.view(b,n_views,-1)
        embs = embs.view(b,n_views,-1)

        return embs,b,n_views,d

    ###################### TODO 真正的ARC
    def forward_f(self, x):
        if self.config.csidataset == 'Widar3.0':
            x_anchor = x[:, 0, :, :, :]                # [b, amp_phase,c, t]
            x_indomain_view = x[:, 1, :, :, :]
            feats_1 = self.feature_extractor(x_anchor) # [b,d]
            feats_2 = self.feature_extractor(x_indomain_view) # [b,d]
            feats_interdomain = self.randomize(feats_2) #[b,d]
            feats = torch.stack([feats_1, feats_2, feats_interdomain], dim=1)  # [b, 3, d]
        else:
            x_1, x_2, x_3 = x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:]
            feats_1 = self.feature_extractor(x_1) # [b,d]
            feats_2 = self.feature_extractor(x_2) # [b,d]
            feats_3 = self.feature_extractor(x_3)
            feats_interdomain2 = self.randomize(feats_2)
            feats_interdomain3 = self.randomize(feats_3)
            feats = torch.stack([feats_1, feats_2, feats_3, feats_interdomain2, feats_interdomain3], dim=1)  # [b, 5,d]
        return feats
        
    def predict(self,x,y):
        # 由于训练的时候是每个view单独求解，为了利用所有的view，在predicting的时候我们使用soft voting的方式
        x=self.decompose(x)
        if self.config.csidataset == 'Widar3.0':
            x_anchor = x[:, 0, :, :, :]                # [b, amp_phase,c, t]
            x_indomain_view = x[:, 1, :, :, :]
            feats_1 = self.feature_extractor(x_anchor) # [b,d]
            feats_2 = self.feature_extractor(x_indomain_view) # [b,d]
            feats = torch.stack([feats_1, feats_2], dim=1)  # [b, 2, d]
        else:
            x_1, x_2, x_3 = x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:]
            feats_1 = self.feature_extractor(x_1) # [b,d]
            feats_2 = self.feature_extractor(x_2) # [b,d]
            feats_3 = self.feature_extractor(x_3)
            feats = torch.stack([feats_1, feats_2, feats_3], dim=1)  # [b, 3, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)
        # 1. cls接在emb后直接计算
        #logits = self.classifier(embs) # [b*n_views, d]
        logits = self.classifier(feats)     
        #embs = embs.view(b,n_views,-1) 
        logits = logits.view(b,n_views,-1)
        probs = F.softmax(logits, dim=-1)      # [b, n_views, num_classes]
        # step2: 平均概率 (soft-voting)
        probs_mean = probs.mean(dim=1)         # [b, num_classes]
        # step3: log + nll_loss
        log_probs = torch.log(probs_mean + 1e-8)  # 防止log(0)

        negative_mask = y < 0
        new_y = y.clone()
        new_y[negative_mask] = torch.abs(new_y[negative_mask] + 1)
        loss = F.nll_loss(log_probs, new_y)
        return probs_mean, loss.sum()

    def decompose(self,x):
        if self.config.csidataset=='Widar3.0': #[bsz,ant,amp+phase,t,c]
            b,a,ap,T,c=x.shape 
            x=x.permute(0,1,2,4,3) # [bsz,ant,amp_phase,sub,time]
            return x
        else:
            # torch.Size([128, 3, 114, 2, 1800]) [bsz,ant,sub,amp_phase,time]
            b,a,c,ap,T=x.shape 
            x=x.permute(0,1,3,2,4)# [bsz,ant,amp_phase,sub,time]
            return x

class UniCrossFi_dg(nn.Module):
    def __init__(self, config,backbone,T_max):

        super(UniCrossFi_dg, self).__init__()
        if config.csidataset == 'Widar3.0':
            data_inchan = 2 
        else:
            data_inchan = 2
        if config.backbone=='ResNet':
            self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],data_inchannel=data_inchan)
        else:
            self.feature_extractor = backbone
        self.config=config
        self.device=torch.device(config.device)

        # neck
        self.neck = nn.Sequential(
                nn.Linear(config.last_dim, config.last_dim ),
                nn.ReLU(),
                nn.Linear(config.last_dim, config.last_dim//4)
            )
        feat_dim = config.last_dim//4
        # 手势分类头
        if config.classify=='linear': #线性分类头 
            self.classifier=nn.Linear(feat_dim,config.num_classes)
        elif config.classify=='nonlinear':
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, feat_dim ),
                nn.ReLU(),
                nn.Linear(feat_dim, config.num_classes)
            )
        else:
             raise ValueError(f"Undefined classifier type: '{config.classify}'. Please choose from ['linear', 'nonlinear'].")
         
        self.optimizer_f=get_optimizer(self.feature_extractor.parameters(),config)
        self.optimizer_c=get_optimizer(self.classifier.parameters(),config)
        self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)
        self.scheduler_c = CosineAnnealingLR(self.optimizer_c, T_max=T_max, eta_min=0)
        self.criterion = SupConLoss(temperature=config.temperature)
    
    def randomize(self, x, eps=1e-5):#torch.Size([128, 512])
        # return x
        ##
        sizes = x.size()
        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)  # [b, c, h*w]

        # 原始 mean/std
        orig_mean = x.mean(-1, keepdim=True)   # [b, c, 1] or [b, 1]
        orig_std = x.std(-1, keepdim=True) + eps

        # 打乱得到新的 mean/std
        idx_swap = torch.randperm(sizes[0], device=x.device)
        new_mean = orig_mean[idx_swap]
        new_std  = orig_std[idx_swap]

        alpha = torch.rand(sizes[0], 1, device=x.device)
        if len(sizes) == 4:
            alpha = alpha.unsqueeze(-1)
        new_mean = alpha * orig_mean + (1 - alpha) * new_mean
        new_std  = alpha * orig_std  + (1 - alpha) * new_std

        # 重新参数化
        x_norm = (x - orig_mean) / orig_std
        x_new = x_norm * new_std + new_mean
    
        return x_new.view(*sizes)

    def update(self,x,y,domain_label,idx,epoch,loader):
        # learn gesture feature
        # print(x.shape)
        # start_time=time.time()
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        feats,logits,emb = self.forward(x) #返回三个部分的输出
        # 计算supcon 和con，这个都是用emb 
        loss_con=self.config.beta*self.criterion(emb, y)+(1-self.config.beta)*self.criterion(emb)

        # 计算 ce loss (对每个 view)
        ce_list = []
        for v in range(logits.size(1)):  # 遍历 n_views
            ce_v = F.cross_entropy(logits[:, v, :], y)
            ce_list.append(ce_v)
        loss_ce = sum(ce_list) / len(ce_list)  # 取平均

        scale = min(self.config.scale_cap, loss_con.detach()/ (loss_ce.detach()+1e-6))   # 保证loss_con不会相对于loss_ce过大
        #scale = 1
        loss = loss_ce + scale * loss_con
        loss.backward() 
        self.optimizer_f.step()
        self.optimizer_c.step()
        self.scheduler_f.step()
        self.scheduler_c.step()
        loss_dict = {'loss_ce':loss_ce.item(),'loss_con':loss_con.item()}
        for v, ce_v in enumerate(ce_list):
            loss_dict[f'loss_ce_view{v}'] = ce_v.item()
        return logits[:,0,:].squeeze(),loss_dict,self.optimizer_f.param_groups[0]['lr'],self.optimizer_c.param_groups[0]['lr']
        
    def forward(self, x):
        x=self.decompose(x)
        feats = self.forward_f(x) # [b, n_views, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)
        # 1. cls接在emb后直接计算
        logits = self.classifier(embs)
        # 2. cls接在 feature 后计算
        # logits = self.classifier(feats)        
        feats = feats.view(b, n_views, -1) # [bsz,nview,d]
        logits = logits.view(b,n_views,-1)
        embs = embs.view(b,n_views,-1)
        return feats,logits,embs
    # ########################### TODO 在这里和后面predict部分计算没有arc的表现，即只使用加高斯噪音的增强###########
    # def add_gaussian_noise(self, x, noise_std=0.01):
    #     """
    #     仅用于消融实验
    #     """
    #     noise = torch.randn_like(x) * noise_std
    #     return x + noise
    # def forward_f(self, x):
    #     b,a,amppha,c,t = x.shape
    #     x = x.reshape(b, -1, c, t)
    #     x_aug = self.add_gaussian_noise(x)
    #     feats_1 = self.feature_extractor(x) # [b,d]
    #     feats_2 = self.feature_extractor(x_aug) # [b,d]
    #     feats = torch.stack([feats_1, feats_2], dim=1)  # [b, 2, d]
    #     return feats
    # def predict(self,x,y):
    #     # 由于训练的时候是每个view单独求解，为了利用所有的view，在predicting的时候我们使用soft voting的方式
    #     x=self.decompose(x)
    #     b,a,amppha,c,t = x.shape
    #     x = x.reshape(b, -1, c, t)
    #     feats = self.feature_extractor(x) # [b,d]
    #     embs = self.neck(feats) # [b*n_views, d] 
    #     embs = F.normalize(embs, dim=1)
    #     # 1. cls接在emb后直接计算
    #     logits = self.classifier(embs) # [b, d]
    #     probs = F.softmax(logits, dim=-1)      # [b, num_classes]
    #     # step3: log + nll_loss
    #     log_probs = torch.log(probs + 1e-8)  # 防止log(0)
    #     loss = F.nll_loss(log_probs, y)
    #     return probs, loss.sum()

    ###################### TODO 真正的ARC
    def forward_f(self, x):
        if self.config.csidataset == 'Widar3.0':
            x_anchor = x[:, 0, :, :, :]                # [b, amp_phase,c, t]
            x_indomain_view = x[:, 1, :, :, :]
            feats_1 = self.feature_extractor(x_anchor) # [b,d]
            feats_2 = self.feature_extractor(x_indomain_view) # [b,d]
            feats_interdomain = self.randomize(feats_2) #[b,d]
            feats = torch.stack([feats_1, feats_2, feats_interdomain], dim=1)  # [b, 3, d]
        else:
            x_1, x_2, x_3 = x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:]
            feats_1 = self.feature_extractor(x_1) # [b,d]
            feats_2 = self.feature_extractor(x_2) # [b,d]
            feats_3 = self.feature_extractor(x_3)
            feats_interdomain2 = self.randomize(feats_2)
            feats_interdomain3 = self.randomize(feats_3)
            feats = torch.stack([feats_1, feats_2, feats_3, feats_interdomain2, feats_interdomain3], dim=1)  # [b, 5,d]
        return feats
        
    def predict(self,x,y):
        # 由于训练的时候是每个view单独求解，为了利用所有的view，在predicting的时候我们使用soft voting的方式
        x=self.decompose(x)
        if self.config.csidataset == 'Widar3.0':
            x_anchor = x[:, 0, :, :, :]                # [b, amp_phase,c, t]
            x_indomain_view = x[:, 1, :, :, :]
            feats_1 = self.feature_extractor(x_anchor) # [b,d]
            feats_2 = self.feature_extractor(x_indomain_view) # [b,d]
            feats = torch.stack([feats_1, feats_2], dim=1)  # [b, 2, d]
        else:
            x_1, x_2, x_3 = x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:]
            feats_1 = self.feature_extractor(x_1) # [b,d]
            feats_2 = self.feature_extractor(x_2) # [b,d]
            feats_3 = self.feature_extractor(x_3)
            feats = torch.stack([feats_1, feats_2, feats_3], dim=1)  # [b, 3, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)
        # 1. cls接在emb后直接计算
        logits = self.classifier(embs) # [b*n_views, d]
        #embs = embs.view(b,n_views,-1) 
        logits = logits.view(b,n_views,-1)
        probs = F.softmax(logits, dim=-1)      # [b, n_views, num_classes]
        # step2: 平均概率 (soft-voting)
        probs_mean = probs.mean(dim=1)         # [b, num_classes]
        # step3: log + nll_loss
        log_probs = torch.log(probs_mean + 1e-8)  # 防止log(0)
        loss = F.nll_loss(log_probs, y)
        return probs_mean, loss.sum()

    def predict4time(self,x,y):
        # 由于训练的时候是每个view单独求解，为了利用所有的view，在predicting的时候我们使用soft voting的方式
        # start_time = time.time()
        x=self.decompose(x)
        if self.config.csidataset == 'Widar3.0':
            x_anchor = x[:, 0, :, :, :]                # [b, amp_phase,c, t]
            x_indomain_view = x[:, 1, :, :, :]
            feats_1 = self.feature_extractor(x_anchor) # [b,d]
            feats_2 = self.feature_extractor(x_indomain_view) # [b,d]
            feats = torch.stack([feats_1, feats_2], dim=1)  # [b, 2, d]
        else:
            x_1, x_2, x_3 = x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:]
            feats_1 = self.feature_extractor(x_1) # [b,d]
            feats_2 = self.feature_extractor(x_2) # [b,d]
            feats_3 = self.feature_extractor(x_3)
            feats = torch.stack([feats_1, feats_2, feats_3], dim=1)  # [b, 3, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)
        # 1. cls接在emb后直接计算
        logits = self.classifier(embs) # [b*n_views, d]
        #embs = embs.view(b,n_views,-1) 
        logits = logits.view(b,n_views,-1)
        probs = F.softmax(logits, dim=-1)      # [b, n_views, num_classes]
        # step2: 平均概率 (soft-voting)
        probs_mean = probs.mean(dim=1)         # [b, num_classes]
        # # step3: log + nll_loss
        return probs_mean

    def get_emb(self,x,y):
        # 由于训练的时候是每个view单独求解，为了利用所有的view，在predicting的时候我们使用soft voting的方式
        x=self.decompose(x)
        if self.config.csidataset == 'Widar3.0':
            x_anchor = x[:, 0, :, :, :]                # [b, amp_phase,c, t]
            x_indomain_view = x[:, 1, :, :, :]
            feats_1 = self.feature_extractor(x_anchor) # [b,d]
            feats_2 = self.feature_extractor(x_indomain_view) # [b,d]
            feats = torch.stack([feats_1, feats_2], dim=1)  # [b, 2, d]
        else:
            x_1, x_2, x_3 = x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:]
            feats_1 = self.feature_extractor(x_1) # [b,d]
            feats_2 = self.feature_extractor(x_2) # [b,d]
            feats_3 = self.feature_extractor(x_3)
            feats = torch.stack([feats_1, feats_2, feats_3], dim=1)  # [b, 3, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)

        # 1. cls接在emb后直接计算
        logits = self.classifier(embs) # [b*n_views, d]
        #embs = embs.view(b,n_views,-1) 
        logits = logits.view(b,n_views,-1)
        probs = F.softmax(logits, dim=-1)      # [b, n_views, num_classes]
        # step2: 平均概率 (soft-voting)
        probs_mean = probs.mean(dim=1)         # [b, num_classes]
        # step3: log + nll_loss
        log_probs = torch.log(probs_mean + 1e-8)  # 防止log(0)
        loss = F.nll_loss(log_probs, y)
        return embs,b,n_views,d

    def decompose(self,x):
        if self.config.csidataset=='Widar3.0': #[bsz,ant,amp+phase,t,c]
            b,a,ap,T,c=x.shape 
            x=x.permute(0,1,2,4,3) # [bsz,ant,amp_phase,sub,time]
            return x
        else:
            # torch.Size([128, 3, 114, 2, 1800]) [bsz,ant,sub,amp_phase,time]
            b,a,c,ap,T=x.shape 
            x=x.permute(0,1,3,2,4)# [bsz,ant,amp_phase,sub,time]
            return x
    

class SimCLR(nn.Module):
    def __init__(self, config,backbone,T_max):

        super(SimCLR, self).__init__()
        if config.csidataset == 'Widar3.0':
            data_inchan = 2 
        else:
            data_inchan = 3
        if config.backbone=='ResNet':
            self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],data_inchannel=data_inchan)
        else:
            self.feature_extractor = backbone
        self.config=config
        self.device=torch.device(config.device)

        # neck
        self.neck = nn.Sequential(
                nn.Linear(config.last_dim, config.last_dim ),
                nn.ReLU(),
                nn.Linear(config.last_dim, config.last_dim//4)
            )
        feat_dim = config.last_dim
        # 手势分类头
        if config.classify=='linear': #线性分类头 
            self.classifier=nn.Linear(feat_dim,config.num_classes)
        elif config.classify=='nonlinear':
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, feat_dim ),
                nn.ReLU(),
                nn.Linear(feat_dim, config.num_classes)
            )
        else:
             raise ValueError(f"Undefined classifier type: '{config.classify}'. Please choose from ['linear', 'nonlinear'].")
         
        self.optimizer_f=get_optimizer(self.feature_extractor.parameters(),config)
        self.optimizer_c=get_optimizer(self.classifier.parameters(),config)
        self.scheduler_f = CosineAnnealingLR(self.optimizer_f, T_max=T_max, eta_min=0)
        self.scheduler_c = CosineAnnealingLR(self.optimizer_c, T_max=T_max, eta_min=0)
        self.criterion = SupConLoss(temperature=config.temperature)
    

    def update(self,x,y,domain_label,idx,epoch,loader):
        # learn gesture feature
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        feats,logits,emb = self.forward(x) #返回三个部分的输出
        # 计算con，这个都是用emb 
        loss=self.criterion(emb)
        loss.backward() 
        self.optimizer_f.step()
        self.optimizer_c.step()
        self.scheduler_f.step()
        self.scheduler_c.step()
        loss_dict = {'loss_con':loss.item()}
        return logits[:,0,:].squeeze(),loss_dict,self.optimizer_f.param_groups[0]['lr'],self.optimizer_c.param_groups[0]['lr']
        
    def forward(self, x):
        x=self.decompose(x)
        feats = self.forward_f(x) # [b, n_views, d]
        b, n_views, d = feats.shape
        feats = feats.view(b*n_views, d) # [b*n_views, d]
        embs = self.neck(feats) # [b*n_views, d] 
        embs = F.normalize(embs, dim=1)
        # 1. simclr 中 cls接在feat后计算
        logits = self.classifier(feats)
        feats = feats.view(b, n_views, -1) # [bsz,nview,d]
        logits = logits.view(b,n_views,-1)
        embs = embs.view(b,n_views,-1)
        return feats,logits,embs
    ###########################  在这里和后面predict部分计算没有arc的表现，即只使用加高斯噪音的增强###########
    def add_gaussian_noise(self, x):
        """
        仅用于消融实验
        """
        noise = torch.randn_like(x) * self.config.noise_std
        return x + noise
    def forward_f(self, x):
        b,a,amppha,c,t = x.shape
        x = x.reshape(b, a, -1, t)
        x_aug = self.add_gaussian_noise(x)
        feats_1 = self.feature_extractor(x) # [b,d]
        feats_2 = self.feature_extractor(x_aug) # [b,d]
        feats = torch.stack([feats_1, feats_2], dim=1)  # [b, 2, d]
        return feats
    def predict(self,x,y):
        # 由于训练的时候是每个view单独求解，为了利用所有的view，在predicting的时候我们使用soft voting的方式
        x=self.decompose(x)
        b,a,amppha,c,t = x.shape
        x = x.reshape(b, a, -1, t)
        feats = self.feature_extractor(x) # [b,d]
        # 1. cls接在feat后直接计算
        logits = self.classifier(feats) # [b, d]
        probs = F.softmax(logits, dim=-1)      # [b, num_classes]
        # step3: log + nll_loss
        log_probs = torch.log(probs + 1e-8)  # 防止log(0)

        negative_mask = y < 0
        new_y = y.clone()
        new_y[negative_mask] = torch.abs(new_y[negative_mask] + 1)
        loss = F.nll_loss(log_probs, new_y)
        return probs, loss.sum()


    def decompose(self,x):
        if self.config.csidataset=='Widar3.0': #[bsz,ant,amp+phase,t,c]
            b,a,ap,T,c=x.shape 
            x=x.permute(0,1,2,4,3) # [bsz,ant,amp_phase,sub,time]
            return x
        else:
            # torch.Size([128, 3, 114, 2, 1800]) [bsz,ant,sub,amp_phase,time]
            b,a,c,ap,T=x.shape 
            x=x.permute(0,1,3,2,4)# [bsz,ant,amp_phase,sub,time]
            return x
