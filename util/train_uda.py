# zzy添加 写的用于UDA的代码

from data.dataset import get_dataset
from models.backbones import Resnet_enc
from .optimizer import get_optimizer
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
    mode,dataset_name,train_loader,test_loader,device,log_wandb,model,total_step,lr,epoch,config,metric_collection
):
    epoch_loss=0
    model.train()
    
    
    n_iter=max(len(train_loader),len(test_loader))
    sample_batch = np.random.randint(low=0, high=n_iter) #采样batch
    batch_iter = 0
    print(f"all iter count is {n_iter}") 
    
    if len(train_loader) > len(test_loader):
        joint_loader =enumerate(zip(train_loader, itertools.cycle(test_loader)))
    else:
        joint_loader =enumerate(zip(itertools.cycle(train_loader), test_loader))
    tbar = tqdm(joint_loader,total=n_iter)
    #开始走训练流程 
    for i ,((src_x,src_y),(trg_x,_)) in tbar:
        # print(i,src_x.shape,src_y.shape,trg_x.shape)
        tbar.set_description(
            "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + config.batch_size)
        )
        batch_iter=batch_iter+config.batch_size
        total_step+=1
        
        
        src_x = src_x.permute(0, 1, 2, 4, 3)
        trg_x = trg_x.permute(0, 1, 2, 4, 3)
        if src_x.shape[0]!=trg_x.shape[0]:
            count=min(src_x.shape[0],trg_x.shape[0])
            src_x=src_x[0:count,:,:,:,:]
            trg_x=trg_x[0:count,:,:,:,:]
            src_y=src_y[0:count]
        b,a,c,n,T=src_x.shape
        src_x=src_x.reshape(b,-1,T).float().to(device)
        trg_x=trg_x.reshape(b,-1,T).float().to(device)
        src_y=src_y.to(device)
        
        preds_train,loss_train,lr_f,lr_c=model.update(src_x,trg_x,src_y)
        epoch_loss+=loss_train
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
        
        
    return log_wandb,model,1,total_step

def val_epoch(
    mode,dataset_name,val_loader,device,log_wandb,model,total_step,lr,epoch,config,metric_collection
):
    model.eval()
    epoch_loss=0
    tbar=tqdm(val_loader)
    batch_iter = 0
    n_iter=len(val_loader)
    for i ,(x,y) in enumerate(tbar):
        tbar.set_description(
            "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + config.batch_size)
        )
        batch_iter=batch_iter+config.batch_size
        total_step+=1
        x = x.permute(0, 1, 2, 4, 3)
        b,a,c,n,T=x.shape
        x=x.reshape(b,-1,T).float().to(device)
        y=y.to(device)
        
        predict_val,loss_val=model.predict(x,y)
        epoch_loss+=loss_val
        
        batch_metrics=metric_collection.forward(predict_val.softmax(dim=-1),y.int())
        log_wandb.log({
            f'{mode} loss': loss_val,
            f'{mode} accuracy': batch_metrics['accuracy'],
            f'{mode} precision': batch_metrics['precision'],
            f'{mode} recall': batch_metrics['recall'],
            f'{mode} f1score': batch_metrics['f1score'],
            'step_val': total_step,
            'epoch': epoch
        })
        
        del x,y
    epoch_metrics = metric_collection.compute() 
    epoch_loss /= n_iter
    for k in epoch_metrics.keys():
        log_wandb.log({f'epoch_{mode}_{str(k)}': epoch_metrics[k],
                       'epoch': epoch})  # log epoch metric
    log_wandb.log({f'epoch_{mode}_loss': epoch_loss,
                   'epoch': epoch})  # log epoch loss
    metric_collection.reset()

    
    return log_wandb,model,1,total_step,epoch_metrics['accuracy']
    
def train(config,log_wandb):

    device = torch.device(config.device)
    set_seed(config.seed)
    ###########################读取数据集###################################
    if config.csidataset=='CSIDA':
        train_dataset, val_dataset, test_dataset = get_dataset(config,None)
    elif config.csidataset=='Widar3.0':
        train_dataset, val_dataset, test_dataset = get_dataset(config,None)
    elif config.csidataset=='SignFi':
        train_dataset, val_dataset, test_dataset = get_dataset(config,None) 
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True) #得到所有的loader
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size , shuffle=False)
    test_loader=DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False)
    
    n_train=len(train_dataset)
    n_val=len(val_dataset)
    n_test=len(test_dataset)
    ###########################读取数据集###################################
    
    
    ###########################得到backbone、以及对应算法###################################
    if config.backbone=='CSIResNet':
        backbone=Resnet_enc(config.inputshape,None) 
    T_max=100 * max(len(train_loader),len(test_loader))
    if config.method=='SourceOnly':
        model=contrast_methods.SourceOnly(config,backbone,T_max) #算法定义成功
    elif config.method=='AdvSKM':
        model=contrast_methods.AdvSKM(config,backbone,T_max)
    else:
        raise ValueError("Invalid method setting.")
    model=model.to(device=device)
    ###########################得到模型###################################
    

    ###########################保存路径+各种utils变量###################################
    check_point_path=f'./checkpoints/uda_zzy/{config.method}/{config.csidataset}{config.cross_domain_type}/{config.target_domain}/'
    os.makedirs(check_point_path, exist_ok=True)
    total_step = 0  # logging step
    total_stop_v=0
    non_improve_epoch=0
    best_val_acc=0.0
    lr = config.lr  # learning rate
    metric_collection = MetricCollection({
        'accuracy': Accuracy(task='multiclass',num_classes=config.num_classes,average='macro').to(device=device),
        'precision': Precision(task='multiclass',num_classes=config.num_classes,average='macro').to(device=device),
        'recall': Recall(task='multiclass',num_classes=config.num_classes,average='macro').to(device=device),
        'f1score': F1Score(task='multiclass',num_classes=config.num_classes,average='macro').to(device=device)
    })
    ###########################保存路径+各种utils变量###################################
     # 日志保存
     
     ###############开始训练#################
    for epoch in range(config.epochs):
        
        #训练过程
        print(epoch)
        log_wandb,model,lr,total_step=train_epoch(mode='train',dataset_name=config.csidataset,train_loader=train_loader,model=model
                                                  ,test_loader=test_loader,device=device,log_wandb=log_wandb,
                                                                    total_step=total_step,lr=lr,epoch=epoch,config=config,metric_collection=metric_collection)
        with torch.no_grad():
            log_wandb,model,lr,total_step_v,this_acc=val_epoch(mode='val',dataset_name=config.csidataset,val_loader=val_loader,model=model,
                                                    device=device,log_wandb=log_wandb,
                                                                        total_step=total_stop_v,lr=lr,epoch=epoch,config=config,metric_collection=metric_collection)
            if this_acc>=best_val_acc:
                non_improve_epoch=0
                best_val_acc=this_acc
                torch.save(model.state_dict(), f'{check_point_path}/checkpoint.pth')
                print(f'{epoch}数据已经保存')
            else:
                non_improve_epoch=non_improve_epoch+1
            if non_improve_epoch > config.early_stop_epoch:
                break
            
    model.load_state_dict(torch.load(f'{check_point_path}/checkpoint.pth', map_location=device))
    result_file = os.path.join(check_point_path, "result.txt")
    with torch.no_grad():
        tbar=tqdm(test_loader)
        for i ,(x,y) in enumerate(tbar):
            
           
            x = x.permute(0, 1, 2, 4, 3)
            b,a,c,n,T=x.shape
            x=x.reshape(b,-1,T).float().to(device)
            y=y.to(device)
            
            predict_test,loss_test=model.predict(x,y)
            batch_metrics=metric_collection.forward(predict_test.softmax(dim=-1),y.int())
            print(batch_metrics)
        epoch_metrics = metric_collection.compute()
        for k in epoch_metrics.keys():
            log_wandb.log({f'epoch_test_{str(k)}': epoch_metrics[k],
                        'epoch': 0})  # log epoch metric 
        with open(result_file, "w") as f:
            for k in epoch_metrics.keys():
                line = f"{k}: {epoch_metrics[k].item():.4f}\n"
                f.write(line)
        metric_collection.reset()
        ###############开始训练#################
   
    
    
    

    
    
   
    