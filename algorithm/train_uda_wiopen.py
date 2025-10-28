# zzy添加 写的用于UDA的代码

from data.dataset import get_dataset
from models.backbones import Resnet_enc
from util.optimizer import get_optimizer
import torch
from sklearn.metrics import accuracy_score, f1_score
import wandb
import os
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
import itertools  
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import contrast_methods
import random
import numpy as np 
import datetime
from tqdm import tqdm
def set_seed(seed: int):
    random.seed(seed)                          # Python 随机模块
    np.random.seed(seed)                       # NumPy 随机模块
    torch.manual_seed(seed)                    # PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)               # PyTorch GPU 随机种子（单卡）
    torch.cuda.manual_seed_all(seed)           # 多卡情况下的随机种子

    # 确保 cudnn 的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def train_epoch(
    mode,dataset_name,train_loader,test_loader,device,log_wandb,model,total_step,lr,epoch,config,metric_collection,train_dataset
):
    epoch_loss=0
    model.train()
    nmax=0.0
    
    n_iter=max(len(train_loader),len(test_loader))
    sample_batch = np.random.randint(low=0, high=n_iter) #采样batch
    batch_iter = 0
    # print(f"all iter count is {n_iter}") 
    
    if len(train_loader) > len(test_loader):
        joint_loader =enumerate(zip(train_loader, itertools.cycle(test_loader)))
    else:
        joint_loader =enumerate(zip(itertools.cycle(train_loader), test_loader))
    # tbar = tqdm(joint_loader,total=n_iter)
    #开始走训练流程 
    # for i ,((src_x,src_y),(trg_x,_)) in tbar:
    for i, ((src_x, src_dfs,src_y,idx), (trg_x,trg_dfs, _,_)) in joint_loader:
        # print(i,src_x.shape,src_y.shape,trg_x.shape)

        batch_iter=batch_iter+config.batch_size
        total_step+=1
        
        if src_x.shape[0]!=trg_x.shape[0] and   config.method=='Wiopen': 
            count=min(src_x.shape[0],trg_x.shape[0])
            src_x=src_x[0:count,:,:,:]
            src_dfs=src_dfs[0:count,:,:,:]
            idx=idx[0:count]
            trg_x=trg_x[0:count,:,:,:]
            trg_dfs=trg_dfs[0:count,:,:,:]
            src_y=src_y[0:count]
            

        # print(src_dfs)

        # print(src_x.shape) # b,3,114,2,1800
        # exit(0)
        
        src_x=src_x.float().to(device)
        src_dfs=src_dfs.float().to(device)
        trg_x=trg_x.float().to(device)
        trg_dfs=trg_dfs.float().to(device)
        src_y=src_y.to(device)
        idx=idx.to(device)
        
  
        preds_train,loss_train,lr_f,lr_c,ncmax=model.update(src_x,trg_x,src_y,epoch,idx,src_dfs)
        nmax=nmax+ncmax
        epoch_loss+=loss_train
        # print(f"epoch_loss{epoch_loss},loss_iter{loss_train}")
        # print(preds_train)
        # print(loss_train)
        batch_metrics=metric_collection.forward(preds_train.softmax(dim=-1),src_y.int())
        log_wandb.log({
            f'{mode} loss': loss_train,
            f'{mode} accuracy': batch_metrics['accuracy'],
            f'{mode} precision': batch_metrics['precision'],
            f'{mode} recall': batch_metrics['recall'],
            f'{mode} f1score': batch_metrics['f1score'],
            f'{mode} lr':lr_c,
            'step': total_step,
            'epoch': epoch
        })
        
        del src_x,trg_x,src_y
    epoch_metrics = metric_collection.compute() 
    epoch_loss /= n_iter
    for k in epoch_metrics.keys():
        log_wandb.log({f'epoch_{mode}_{str(k)}': epoch_metrics[k],
                       'epoch': epoch})  # log epoch metric
    log_wandb.log({f'epoch_{mode}_loss': epoch_loss,
                   'epoch': epoch})  # log epoch loss
    metric_collection.reset()
    print(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}")
        
        
    return log_wandb,model,1,total_step,nmax

def val_epoch(
    mode,dataset_name,val_loader,device,log_wandb,model,total_step,lr,epoch,config,metric_collection,nmax,train_loader,t=0.5
):
    model.eval()

    net=model.network
    lemniscate = model.lemniscate 

    net.eval()
    total = 0
    total1 = 0
    testsize = val_loader.dataset.__len__()
    trainsize = train_loader.dataset.__len__()
    trainFeatures = lemniscate.memory.t() 
    trainLabels=torch.LongTensor(train_loader.dataset.source_label_list).cuda()
    top1 = 0.
    top5 = 0.
    top1un = 0.
    top5un = 0.
    import time
    end = time.time()
    prediction = []
    target = []
    C=trainLabels.max()+1
    K=50


    with torch.no_grad():
        #生成一个 (50,C)的向量
        retrieval_one_hot = torch.zeros(K, C).cuda()
        #生成一个(n_data,C)的向量 
        retrieval_one_hotall = torch.zeros(trainsize, C).cuda()
        p = 0
        #targets就是labels
        for batch_idx, (src_x, src_dfs,src_y,idx) in enumerate(val_loader):

            targets = src_y.to(device)
            src_x = src_x.float().to(device)
            batchSize = src_x.shape[0]
            idx=idx.to(device)
            _,features,output = net(src_x)
            outputs = lemniscate(features, idx) #计算出了新的 相似度矩阵 



            dist = torch.mm(features, trainFeatures) #相似度 矩阵  
            # print(dist.shape)
            # exit(0)

            #yd是距离 yi是索引 也就是返回最相似的K个样本 
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
       
            #（b,n_data） 有b行 每一行都是 样本个数
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            
            #取出来 距离最近的 50个对应的标签 结果是(b,50)
            retrieval = torch.gather(candidates, 1, yi)
          

            
            # 重新赋值  retrieval_one_hot
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            #按照retrieval 设置onehot batchSize * K 代表给一个batch当中的每个样本 K个近邻机会 给他幅值 
        
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            
            # exit(0)
            
            #对距离做变换 
            yd_transform = yd.clone().div_(t).exp_()
            #相当于做了个加权 距离越近这个概率越大 
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            #排序 得到索引 看是哪个类别 
            _, predictions = probs.sort(1, True)

            # 
     
            correct = predictions.eq(targets.data.view(-1,1))
            # prediction.extend(prela)
            target.extend(targets.data.view(-1,1).cpu().numpy())
  

            top1 = top1 + correct.narrow(1,0,1).sum().item() #narrow切片操作 第一个维度 从0开始 找到前1个 
            # top5 = top5 + correct.narrow(1,0,3).sum().item()

            total += targets.size(0)
        # print(f'top1:{top1},total:{total}')
        
        accuracy=top1 / total
        print(f'top1: {top1}, total: {total}, accuracy: {top1 / total * 100:.2f}%')


    
    return log_wandb,model,1,total_step,accuracy
    
def train(config,log_wandb):

    device = torch.device(config.device)
    
    ###########################读取数据集###################################
    if config.csidataset=='CSIDA':
        train_dataset, val_dataset, test_dataset = get_dataset(config,None)
    elif config.csidataset=='Widar3.0':
        train_dataset, val_dataset, test_dataset = get_dataset(config,None)
    else:
        raise ValueError(f"{config.csidataset} is not supported")


    
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=15) #得到所有的loader
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size , shuffle=False,num_workers=15)
    test_loader=DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False,num_workers=15)
    
    n_train=len(train_dataset)
    n_val=len(val_dataset)
    n_test=len(test_dataset)
    set_seed(config.seed)
    ###########################读取数据集###################################
    
    
    ###########################得到backbone、以及对应算法###################################
    if config.backbone=='CSIResNet':
        backbone=Resnet_enc(config.inputshape,None) 
    elif config.backbone=='ResNet2d':
        pass
    else:
        raise ValueError(f"{config.backbone} is not supported")
    T_max=100 * max(len(train_loader),len(test_loader))
    if config.method=='SourceOnly':
        model=contrast_methods.SourceOnly(config,backbone,T_max) #算法定义成功
    elif config.method=='AdvSKM':
        model=contrast_methods.AdvSKM(config,backbone,T_max)
    elif config.method=='CoTMix':
        model=contrast_methods.CoTMix(config,backbone,T_max)
    elif config.method=='WiGRUNT':
        model=contrast_methods.WiGRUNT(config,backbone,T_max)
    elif config.method=='WiSDA':
        model=contrast_methods.WiSDA(config,backbone,T_max)
    elif config.method=='Wiopen':
        model=contrast_methods.Wiopen(config,backbone,T_max,n_train,train_dataset)
        
    else:
        raise ValueError("Invalid method setting.") #算法定义成功
    model=model.to(device=device)
    ###########################得到模型###################################
    

    ###########################保存路径+各种utils变量###################################
    check_point_path=f'./checkpoints/uda_zzy/{config.method}/{config.csidataset}/{config.cross_domain_type}/{config.target_domain}/'
    os.makedirs(check_point_path, exist_ok=True)
    total_step = 0  # logging step
    total_stop_v=0
    non_improve_epoch=0
    best_val_acc=0.0
    lr = config.lr  # learning rate
    metric_collection = MetricCollection({
        'accuracy': Accuracy(task='multiclass',num_classes=config.num_classes,average='micro').to(device=device),
        'precision': Precision(task='multiclass',num_classes=config.num_classes,average='macro').to(device=device),
        'recall': Recall(task='multiclass',num_classes=config.num_classes,average='macro').to(device=device),
        'f1score': F1Score(task='multiclass',num_classes=config.num_classes,average='macro').to(device=device)
    })
    ###########################保存路径+各种utils变量###################################
     # 日志保存
     
     ###############开始训练#################
    for epoch in range(config.epochs):
        
        #训练过程
        # print(epoch)
        log_wandb,model,lr,total_step,nmax=train_epoch(mode='train',dataset_name=config.csidataset,train_loader=train_loader,model=model
                                                  ,test_loader=test_loader,device=device,log_wandb=log_wandb,
                                                                    total_step=total_step,lr=lr,epoch=epoch,config=config,metric_collection=metric_collection,train_dataset=train_dataset)
        with torch.no_grad():
            log_wandb,model,lr,total_step_v,this_acc=val_epoch(mode='val',dataset_name=config.csidataset,val_loader=val_loader,model=model,
                                                    device=device,log_wandb=log_wandb,
                                                                        total_step=total_stop_v,lr=lr,epoch=epoch,config=config,metric_collection=metric_collection,nmax=nmax,train_loader=train_loader,t=config.wiopen_temperature
                                                                        )
            if this_acc>=best_val_acc:
                non_improve_epoch=0
                best_val_acc=this_acc
                torch.save(model.state_dict(), f'{check_point_path}/checkpoint_seed{config.seed}.pth')
                # print(f'{epoch}数据已经保存')
            else:
                non_improve_epoch=non_improve_epoch+1
            if non_improve_epoch > config.early_stop_epoch:
                break
            
    model.load_state_dict(torch.load(f'{check_point_path}/checkpoint_seed{config.seed}.pth', map_location=device))
    result_file = os.path.join(check_point_path, "result_multiseed.txt")

    model.eval()
    
 

    net=model.network
    lemniscate = model.lemniscate 

    net.eval()
    total = 0
    total1 = 0
    testsize = test_loader.dataset.__len__()
    trainsize = train_loader.dataset.__len__()
    trainFeatures = lemniscate.memory.t() 
    trainLabels=torch.LongTensor(train_loader.dataset.source_label_list).cuda()
    top1 = 0.
    top5 = 0.
    top1un = 0.
    top5un = 0.
    import time
    end = time.time()
    prediction = []
    target = []
    C=trainLabels.max()+1
    K=50


    with torch.no_grad():
        #生成一个 (50,C)的向量
        retrieval_one_hot = torch.zeros(K, C).cuda()
        #生成一个(n_data,C)的向量 
        retrieval_one_hotall = torch.zeros(trainsize, C).cuda()
        p = 0
        #targets就是labels
        for batch_idx, (src_x, src_dfs,src_y,idx) in enumerate(test_loader):

            targets = src_y.to(device)
            src_x = src_x.float().to(device)
            batchSize = src_x.shape[0]
            idx=idx.to(device)
            _,features,output = net(src_x)
            outputs = lemniscate(features, idx) #计算出了新的 相似度矩阵 



            dist = torch.mm(features, trainFeatures) #相似度 矩阵  
            # print(dist.shape)
            # exit(0)

            #yd是距离 yi是索引 也就是返回最相似的K个样本 
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
       
            #（b,n_data） 有b行 每一行都是 样本个数
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            
            #取出来 距离最近的 50个对应的标签 结果是(b,50)
            retrieval = torch.gather(candidates, 1, yi)
          

            
            # 重新赋值  retrieval_one_hot
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            #按照retrieval 设置onehot batchSize * K 代表给一个batch当中的每个样本 K个近邻机会 给他幅值 
        
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            
            # exit(0)
            
            #对距离做变换 
            yd_transform = yd.clone().div_(config.wiopen_temperature).exp_()
            #相当于做了个加权 距离越近这个概率越大 
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            #排序 得到索引 看是哪个类别 
            _, predictions = probs.sort(1, True)

            # 
     
            correct = predictions.eq(targets.data.view(-1,1))
            # prediction.extend(prela)
            target.extend(targets.data.view(-1,1).cpu().numpy())
  

            top1 = top1 + correct.narrow(1,0,1).sum().item() #narrow切片操作 第一个维度 从0开始 找到前1个 
            # top5 = top5 + correct.narrow(1,0,3).sum().item()

            total += targets.size(0)
        # print(f'top1:{top1},total:{total}')
        
        accuracy=top1 / total
        print(f'top1: {top1}, total: {total}, accuracy: {top1 / total * 100:.2f}%')
    
        with open(result_file, "a") as f:
            # 写入实验环境信息
            f.write(f"--- Experiment Environment ---\n")
            f.write(f"Time: 1\n")
            f.write(f"Learning Rate (config.lr): {config.lr}\n")
            f.write(f"Weight Decay (config.weight_decay): {config.weight_decay}\n")
            f.write(f"Seed (config.seed): {config.seed}\n")

            # 写入 best_val_acc 信息
            f.write(f"\nBest Validation Accuracy (best_val_acc): {best_val_acc:.4f}  # 训练过程中验证集上出现的最高准确率\n")
            f.write(f"\ntest_acc: {accuracy:.4f}  # 训练过程中验证集上出现的最高准确率\n")


            # 写入所有 epoch_metrics 中的内容

    

            # 末尾添加分隔符
            f.write("\n" + "=" * 40 + "\n\n")
            
            print(f"test manual accuracy: {accuracy:.4f}")
            log_wandb.log(
                {'test_accuracy_manual':accuracy,}
            )
        ###############开始训练#################
   
    
    
    

    
    
   
    