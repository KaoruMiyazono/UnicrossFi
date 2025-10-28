from data.CSISampler import DomainBalancedBatchSampler
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
import time
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
   
    batch_iter = 0
    # print(f"all iter count is {n_iter}") 
    
    # start_time = time.time()
    for i, (x, y,domain,domain_str,idx) in enumerate(train_loader):
        # print(i,src_x.shape,src_y.shape,trg_x.shape)

        batch_iter=batch_iter+config.batch_size
        total_step+=1

        bs=len(x)
        x = x.permute(0, 1, 2, 4, 3)


        x=x.float().to(device)
        y=y.to(device)

        domain=domain.to(device)
        idx=idx.to(device)
        
        preds_train,loss_train,lr_f,lr_c=model.update(x,y,domain,idx,epoch,train_loader)
        log_data = {}
        if isinstance(loss_train, dict):
            # 记录总loss并累加
            epoch_loss += loss_train.get("total", sum(loss_train.values()))
            for k, v in loss_train.items():
                log_data[f"{mode} loss/{k}"] = v
        else:
            epoch_loss += loss_train
            log_data[f"{mode} loss"] = loss_train

        # 因为在dataset中，对于无标签数据我们进行y = -y-1的操作
        negative_mask = y < 0
        y[negative_mask] = torch.abs(y[negative_mask] + 1)
        # 计算 batch metric
        batch_metrics = metric_collection.forward(preds_train[0:bs, :].softmax(dim=-1), y.int())
        for k, v in batch_metrics.items():
            log_data[f"{mode} {k}"] = v

        # 记录学习率和 step/epoch
        log_data[f"{mode} lr"] = lr_c
        log_data["step"] = total_step
        log_data["epoch"] = epoch

        log_wandb.log(log_data)

        del x, y, domain, domain_str, idx
    # end_time = time.time()
    # print(f"训练时间 {end_time-start_time} s")
    # exit(0)
    epoch_metrics = metric_collection.compute() 
    epoch_loss /= n_iter
    for k in epoch_metrics.keys():
        log_wandb.log({f'epoch_{mode}_{str(k)}': epoch_metrics[k],
                       'epoch': epoch})  # log epoch metric
    log_wandb.log({f'epoch_{mode}_loss': epoch_loss,
                   'epoch': epoch})  # log epoch loss
    metric_collection.reset()
    print(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}")
        
    return log_wandb,model,1,total_step

def val_epoch(
    mode,dataset_name,val_loader,device,log_wandb,model,total_step,lr,epoch,config,metric_collection
):
    model.eval()
    epoch_loss=0

    batch_iter = 0
    n_iter=len(val_loader)
    predict_val_list=[]
    label_list=[]
    
    for i ,(x,y,domain,domain_str,label) in enumerate(val_loader):

        batch_iter=batch_iter+config.batch_size
        total_step+=1

        x = x.permute(0, 1, 2, 4, 3)

        x=x.float().to(device)
        y=y.to(device) 
        predict_val,loss_val=model.predict(x,y)

        # 因为在dataset中，对于无标签数据我们进行y = -y-1的操作
        negative_mask = y < 0
        y[negative_mask] = torch.abs(y[negative_mask] + 1)

        epoch_loss+=loss_val
        predict_val_list.append(predict_val)
        label_list.append(y.int())
        
        batch_metrics=metric_collection.forward(predict_val.softmax(dim=-1), y.int())
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
    
    print(f"Epoch {epoch} validation completed. Average loss: {epoch_loss:.4f}, Accuracy: {epoch_metrics['accuracy']:.4f}")
    acc_manual = (torch.cat(predict_val_list, dim=0).argmax(dim=-1) == torch.cat(label_list, dim=0)).float().mean().item()
    print(f"Epoch {epoch} manual accuracy: {acc_manual:.4f}")


    
    return log_wandb,model,1,total_step,epoch_metrics['accuracy']
    
def train(config,log_wandb,optuna_trial=None):

    device = torch.device(config.device)
    
    ###########################读取数据集###################################
    if config.csidataset=='CSIDA':
        train_dataset, val_dataset, test_dataset = get_dataset(config,None)
    elif config.csidataset=='Widar3.0':
        train_dataset, val_dataset, test_dataset = get_dataset(config,None)

    else:
        raise ValueError(f"{config.csidataset} is not supported")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=15) 
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size , shuffle=False,num_workers=15)
    test_loader=DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False,num_workers=15)
    
    n_train=len(train_dataset)
    n_val=len(val_dataset)
    n_test=len(test_dataset)
    set_seed(config.seed)
    ###########################读取数据集###################################
    config.num_domains = len(train_dataset.get_all_domain_labels())  # 获取源域数据集的域标签数量
    print(f"🧠 领域数量 {config.num_domains}")  
    
    ###########################得到backbone以及对应算法###################################
    if config.backbone=='CSIResNet':
        backbone=Resnet_enc(config.inputshape,None) 
    elif config.backbone=='ResNet':
        backbone=None
        pass
    else:
        raise ValueError(f"{config.backbone} is not supported")
    T_max=100 * max(len(train_loader),len(test_loader))
    if config.method=='SourceOnly':
        backbone='ResNet'
        model=contrast_methods.SourceOnly(config,backbone,T_max) 
    elif config.method=='ERM':
        model=contrast_methods.ERM(config,backbone,T_max)    
    elif config.method=='SimCLR':
        model=contrast_methods.SimCLR(config,backbone,T_max)    
    elif config.method=='UniCrossFi_dg':
        model=contrast_methods.UniCrossFi_dg(config,backbone,T_max)         
    elif config.method=='UniCrossFi_semidg':
        model=contrast_methods.UniCrossFi_semidg(config,backbone,T_max)    
    elif config.method=='WiSR':
        model=contrast_methods.WiSR(config,backbone,T_max) 
    else:
        raise ValueError("Invalid method setting.") 
    model=model.to(device=device)

    check_point_path=f'{config.output_dir}/{config.method}/{config.csidataset}/{config.cross_domain_type}/{config.target_domain}/'
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

    for epoch in range(config.epochs):
        
        #训练过程
        log_wandb,model,lr,total_step=train_epoch(mode='train',dataset_name=config.csidataset,train_loader=train_loader,model=model
                                                  ,test_loader=test_loader,device=device,log_wandb=log_wandb,
                                                                    total_step=total_step,lr=lr,epoch=epoch,config=config,metric_collection=metric_collection)
        with torch.no_grad():

            log_wandb,model,lr,total_step_v,this_acc=val_epoch(mode='val',dataset_name=config.csidataset,val_loader=val_loader,model=model,
                                                    device=device,log_wandb=log_wandb,
                                                                        total_step=total_stop_v,lr=lr,epoch=epoch,config=config,metric_collection=metric_collection)
            if optuna_trial is not None:
                optuna_trial.report(this_acc, step=epoch)
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            if this_acc>=best_val_acc:
                non_improve_epoch=0
                best_val_acc=this_acc
                torch.save(model.state_dict(), f'{check_point_path}/checkpoint_seed{config.seed}.pth')
            else:
                non_improve_epoch=non_improve_epoch+1
            if non_improve_epoch > config.early_stop_epoch:
                break
            
    model.load_state_dict(torch.load(f'{check_point_path}/checkpoint_seed{config.seed}.pth', map_location=device))
    result_file = os.path.join(check_point_path, "result_multiseed.txt")
    model.eval()
    predict_test_list=[]
    label_list=[]
    # print(model.training)

    with torch.no_grad():
        for i ,(x,y,domain,domain_str,label) in enumerate(test_loader):
            

            x = x.permute(0, 1, 2, 4, 3)
            
            x=x.float().to(device)
            y=y.to(device)

            predict_test,loss_test=model.predict(x,y)

            # 因为在dataset中，对于无标签数据我们进行y = -y-1的操作
            negative_mask = y < 0
            y[negative_mask] = torch.abs(y[negative_mask] + 1)
            predict_test_list.append(predict_test)
            label_list.append(torch.abs(y).int())
            batch_metrics=metric_collection.forward(predict_test.softmax(dim=-1),y.int())
        epoch_metrics = metric_collection.compute()
        for k in epoch_metrics.keys():
            log_wandb.log({f'epoch_test_{str(k)}': epoch_metrics[k],
                        'epoch': 0})  
       
        metric_collection.reset()
        acc_manual = (torch.cat(predict_test_list, dim=0).argmax(dim=-1) == torch.cat(label_list, dim=0)).float().mean().item()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(result_file, "a",encoding="utf-8") as f:
            # 写入实验环境信息
            f.write(f"--- Experiment Environment ---\n")
            f.write(f"Time: {now}\n")
            f.write(f"Learning Rate (config.lr): {config.lr}\n")
            f.write(f"Weight Decay (config.weight_decay): {config.weight_decay}\n")
            f.write(f"Seed (config.seed): {config.seed}\n")
            f.write(f"Ratio (config.ratio): {config.ratio}\n")

            # 写入 best_val_acc 信息
            f.write(f"\nBest Validation Accuracy (best_val_acc): {best_val_acc:.4f}  # 训练过程中验证集上出现的最高准确率\n")

            # 写入 acc_manual 和 epoch_metrics['accuracy'] 的信息
            f.write(f"Manual Accuracy (acc_manual): {acc_manual:.4f}  # 自定义手动计算得到的准确率\n")
            f.write(f"Logged Accuracy (epoch_metrics['accuracy']): {epoch_metrics['accuracy'].item():.4f}  # 自动日志记录的准确率指标\n")

            # 写入所有 epoch_metrics 中的内容
            f.write(f"\n--- Epoch Metrics ---\n")
            for k in epoch_metrics.keys():
                line = f"{k}: {epoch_metrics[k].item():.4f}\n"
                f.write(line)

            # 末尾添加分隔符
            f.write("\n" + "=" * 40 + "\n\n")
        print(f"test manual accuracy: {acc_manual:.4f}")
        log_wandb.log(
            {'test_accuracy_manual':acc_manual,}
        )
        ###############开始训练#################
    return best_val_acc
