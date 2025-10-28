import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import torchvision.transforms as transforms
import pickle
import math
import torchvision.models as models
from torch.autograd.function import Function


import torch
from torch.autograd import Variable
from torch import nn

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

    
class Res18Featured(nn.Module):
    def __init__(self, pretrained = True, num_classes = 6, drop_rate = 0):
        super(Res18Featured, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)       
        self.features = nn.Sequential(*list(resnet.children())[:-2]) # after avgpool 512x1
        self.feavg = nn.Sequential(*list(resnet.children())[-2:-1])
        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
   
        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())
        self.l2norm = Normalize(2)
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3 ,1, 1),   # [, 256, 7, 7]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 3, 2, 1, 1),   # [, 128, 14, 14]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),    # [, 64, 28, 28]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),      # [, 32, 56, 56]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),      # [, 32, 112, 112]
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),  # [, 16, 224, 224]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 1, 1),         # [, 3, 224, 224]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x1 = self.feavg(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.l2norm(x1)
        out = self.fc(x1)
        rec = self.Decoder(x)
        #print(rec.size())
        return rec,x1,out
    
import torch
from torch.autograd import Function
from torch import nn
import math

class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T) # batchSize * N
        
        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5): #特征长度 样本个数 
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.tensor([T, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out


import torch
from torch import nn
from torch.autograd import Function
import math

eps = 1e-8

class NCACrossEntropy(nn.Module): 
    ''' \sum_{j=C} log(p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    '''
    def __init__(self, labels, margin=0):
        super(NCACrossEntropy, self).__init__()
        self.register_buffer('labels', torch.LongTensor(labels.size(0)))
        self.labels = labels
        self.margin = margin

    def forward(self, x, features, indexes): #传进去的是相似度(b,n_data),特征(b,512) 和 这一个batch样本位置的集合(b) 
        
        #batch_size 代表batch_size n代表样本个数  
        batchSize = x.size(0)
        n = x.size(1)
        exp = torch.exp(x)

        # print(f"batchSize:{batchSize},n:{n},indexes_shape:{indexes.shape}")
        
        # labels for currect batch

        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1) #取出来 这个batch所有的label   
        #for i in range(1,n):
    
        """
        这个 same先对 y做了repeat 也就是变成了 一个 (b,n)的向量 每一行都是一样的 代表当前批次对应位置的标签 比如一开始标签是1,2,3,4,一共有6个,那么就会变成(4,6)矩阵
        假设标签是[1,2,3,4,2,3]
        1,1,1,1,1,1
        2,2,2,2,2,2
        3,3,3,3,3,3
        4,4,4,4,4,4
        然后.eq操作 把它变成一个布尔矩阵 每一行代表 该行所代表的标签和大标签集合哪个相同,比如以下矩阵可以变为如下 
        1,0,0,0,0,0
        0,1,0,0,1,0
        0,0,1,0,0,1
        0,0,0,1,0,0
        """
        
        same = y.repeat(1, n).eq_(self.labels)

       # self prob exclusion, hack with memory for effeciency 把每i行当中 index[i]置0 也就是自己和自己的相似度置0
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        #和当前样本不同类的通过相乘置0也就是没有权重,只有同类才有权重  最后sum后结果是每个样本和自己的正样本的相似度和 
        p = torch.mul(exp, same.float()).sum(dim=1)
        Z = exp.sum(dim=1) #Z 当前样本和所有样本的相似度和 

        Z_exclude = Z - p #和负样本的相似度和
        p = p.div(math.exp(self.margin))
        Z = Z_exclude + p

        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))

        loss = prob_masked.log().sum(0) #拉近了同类别之间的距离

        return - loss / batchSize, p.min(), p.mean()

class NCA(nn.Module): 
    ''' \sum_{j=C} log(p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    '''
    def __init__(self, labels, margin=0):
        super(NCA, self).__init__()
        self.register_buffer('labels', torch.LongTensor(labels.size(0)))
        self.labels = labels
        self.margin = margin

    def forward(self, x, features, indexes):
        batchSize = x.size(0)
        n = x.size(1)
        exp = torch.exp(x)
        
        # labels for currect batch
        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1) 
        #for i in range(1,n):
        same = y.repeat(1, n).eq_(self.labels)

       # self prob exclusion, hack with memory for effeciency
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(exp, same.float()).sum(dim=1)

        return p
    
import torch
import time
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import manifold,datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    classification_report, precision_recall_fscore_support, roc_auc_score

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def NN(epoch, net, lemniscate, trainloader, testloader, testloaderun, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            _,features,_ = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (dfs, inputs, targets, indexes) in enumerate(testloader,testloaderun):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
            
            cls_time.update(time.time() - end)
            end = time.time()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(
                  total, testsize, correct*100./total, net_time=net_time, cls_time=cls_time))

    return correct/total

def _area_under_roc(label, predict, prediction_scores: np.array = None, multi_class='ovo') -> float:
        label = label
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(np.array(predict).reshape(-1, 1))
        # assert prediction_scores.shape == true_scores.shape
        return roc_auc_score(true_scores, prediction_scores, multi_class=multi_class)

def kNN(epoch, nmax, net, lemniscate, trainloader, testloader, testloaderun, K, sigma, recompute_memory=0): 
    #K写死是50
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    total1 = 0
    testsize = testloader.dataset.__len__()
    trainsize = trainloader.dataset.__len__()

    #feature_dim,n_data
    trainFeatures = lemniscate.memory.t() 
    #得到 trian_label
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.img_labels).cuda()
        
    #得到 +2个类别 
    C = trainLabels.max() + 2

    #不想用memory_bank里面的特征
    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.label).cuda()
        trainloader.dataset.transform = transform_bak
    
    top1 = 0.
    top5 = 0.
    top1un = 0.
    top5un = 0.
    end = time.time()
    prediction = []
    target = []
    with torch.no_grad():
        #生成一个 (50,C)的向量
        
        #生成一个(n_data,C)的向量 
        retrieval_one_hotall = torch.zeros(trainsize, C).cuda()
        p = 0
        #targets就是labels
        for batch_idx, (_,inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda()
            batchSize = inputs.size(0)
            retrieval_one_hot = torch.zeros(batchSize * K, C).cuda()
            _,features,output = net(inputs)
            outputs = lemniscate(features, indexes) #计算出了新的 相似度矩阵 
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures) #相似度 矩阵  

            #yd是距离 yi是索引 也就是返回最相似的K个样本 
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            #（b,n_data） 有b行 每一行都是 样本个数
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            
            #取出来 距离最近的 50个对应的标签 结果是(b,50)
            retrieval = torch.gather(candidates, 1, yi)

            
            # 重新赋值  retrieval_one_hot
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            print(retrieval_one_hot.shape)
            exit(0)
            #按照retrieval 设置onehot batchSize * K 代表给一个batch当中的每个样本 K个近邻机会 给他幅值 
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            
            #对距离做变换 
            yd_transform = yd.clone().div_(sigma).exp_()
            #相当于做了个加权 距离越近这个概率越大 
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            #排序 得到索引 看是哪个类别 
            _, predictions = probs.sort(1, True)

            # 
            ydall, yiall = dist.topk(trainsize, dim=1, largest=True, sorted=True)
            candidatesall = trainLabels.view(1,-1).expand(batchSize, -1)
            retrievalall = torch.gather(candidatesall, 1, yiall)
            retrieval_one_hotall.resize_(batchSize * trainsize, C).zero_()
            retrieval_one_hotall.scatter_(1, retrievalall.view(-1, 1), 1)
            yd_transformall = ydall.clone().div_(sigma).exp_()
            probsall = torch.sum(torch.mul(retrieval_one_hotall.view(batchSize, -1 , C), yd_transformall.view(batchSize, -1, 1)), 1)
            pall, predictions1all = probsall.sort(1, True)

            # Find which predictions match the target
            prela = predictions.cpu().numpy()[:,0]
            for i in range(0,batchSize):
                p = p + (pall[i,0]-pall[i,1]-pall[i,2])
                if pall[i,0]<nmax*2:
                    prela[i]=C-1
                    predictions[i,0]=C-1
            correct = predictions.eq(targets.data.view(-1,1))
            prediction.extend(prela)
            target.extend(targets.data.view(-1,1).cpu().numpy())
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item() #narrow切片操作 第一个维度 从0开始 找到前3个 
            top5 = top5 + correct.narrow(1,0,3).sum().item()

            total += targets.size(0)
        retrieval_one_hot = torch.zeros(K, C).cuda()
        retrieval_one_hotall = torch.zeros(trainsize, C).cuda()
        for batch_idx, (_,inputs, targets, indexes) in enumerate(testloaderun):
            end = time.time()
            targets1 = targets.cuda()
            batchSize = inputs.size(0)
            _,features,output = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions1 = probs.sort(1, True)

            ydall, yiall = dist.topk(trainsize, dim=1, largest=True, sorted=True)
            candidatesall = trainLabels.view(1,-1).expand(batchSize, -1)
            retrievalall = torch.gather(candidatesall, 1, yiall)
            retrieval_one_hotall.resize_(batchSize * trainsize, C).zero_()
            retrieval_one_hotall.scatter_(1, retrievalall.view(-1, 1), 1)
            yd_transformall = ydall.clone().div_(sigma).exp_()
            probsall = torch.sum(torch.mul(retrieval_one_hotall.view(batchSize, -1 , C), yd_transformall.view(batchSize, -1, 1)), 1)
            pall, predictions1all = probsall.sort(1, True)

            prela1 = predictions1.cpu().numpy()[:,0]
            for i in range(0,batchSize):
                if pall[i,0]<nmax*2:
                    prela1[i]=C-1
                    predictions1[i,0]=C-1
            correct1 = predictions1.eq(targets1.data.view(-1,1))
            prediction.extend(prela1)
            target.extend(targets1.data.view(-1,1).cpu().numpy())
            cls_time.update(time.time() - end)

            top1un = top1un + correct1.narrow(1,0,1).sum().item()
            top5un = top5un + correct1.narrow(1,0,3).sum().item()

            total1 += targets1.size(0)
        for i in range(len(target)):
            if target[i] < (C-1).cpu().numpy():
                target[i]=0
            else:
                target[i]=1
        for i in range(len(target)):
            if prediction[i] < (C-1).cpu().numpy():
                prediction[i]=0
            else:
                prediction[i]=1
        auroc = _area_under_roc(target, prediction)

        print('Test [{}/{}]\t'
                'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                'Top1: {:.2f}  Top3: {:.2f}\t'
                'Top1un: {:.2f}  Top3un: {:.2f}\t'
                'AUROC: {:.2f}'.format(
                total, testsize, top1*100./total, top5*100./total, top1un*100./total1, top5un*100./total1, auroc, net_time=net_time, cls_time=cls_time))

    print(top1*100./total)

    return top1/total

