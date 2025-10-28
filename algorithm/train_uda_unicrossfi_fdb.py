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
    mode,dataset_name,train_loader,test_loader,device,log_wandb,model,total_step,lr,epoch,config,metric_collection):
    epoch_loss=0
    model.train()
    
    loss_ce=0.0
    loss_con=0.0
    loss_ucon=0.0
    loss_supcon=0.0
    loss_src_supcon=0.0
    loss_src_ucon=0.0
    loss_tgt_supcon=0.0
    loss_tgt_ucon=0.0
    loss_fbc = 0.0
    
    n_iter=max(len(train_loader),len(test_loader))
    sample_batch = np.random.randint(low=0, high=n_iter) #采样batch
    batch_iter = 0
    # print(f"all iter count is {n_iter}") 
    
    if len(train_loader) > len(test_loader):
        joint_loader =enumerate(zip(train_loader, itertools.cycle(test_loader)))
    else:
        joint_loader =enumerate(zip(itertools.cycle(train_loader), test_loader))
    # tbar = tqdm(joint_loader,total=n_iter)
    # joint_loader =enumerate(zip(train_loader, train_loader))

    #开始走训练流程 
    # TODO 把source and target的dataloader都换成会返回Domain label的,且注意target的domain label应当要和source 的区分开来
    # for i ,((src_x,src_y),(trg_x,_)) in tbar:
    for i, ((src_x, src_y,_,_,_), (trg_x, _,_,_,_)) in joint_loader:

        batch_iter=batch_iter+config.batch_size
        total_step+=1
        
        
        src_x = src_x.permute(0, 1, 2, 4, 3)
        trg_x = trg_x.permute(0, 1, 2, 4, 3)
        # print(f'外面{src_x.shape}')
            
        if src_x.shape[0]!=trg_x.shape[0]:
            count=min(src_x.shape[0],trg_x.shape[0])
            src_x=src_x[0:count,:,:,:,:]
            trg_x=trg_x[0:count,:,:,:,:]
            src_y=src_y[0:count]
    


        # print(src_x.shape) # b,3,114,2,1800
        # exit(0)
        
        src_x=src_x.float().to(device)
        trg_x=trg_x.float().to(device)
        src_y=src_y.to(device)
        
  
        preds_train,loss_train,lr_f,lr_c=model.update(src_x,trg_x,src_y,None,None,epoch,None)
        epoch_loss+=loss_train['loss_ce']+loss_train['loss_con']
        loss_ce+=loss_train['loss_ce']
        loss_con+=loss_train['loss_con']
        loss_ucon+=loss_train.get('loss_ucon', 0)
        loss_supcon+=loss_train.get('loss_supcon', 0)
        loss_src_supcon+=loss_train.get('loss_src_supcon', 0)
        loss_src_ucon += loss_train.get('loss_src_ucon', 0)
        loss_tgt_supcon+=loss_train.get('loss_tgt_supcon', 0)
        loss_tgt_ucon += loss_train.get('loss_tgt_ucon', 0)
        loss_fbc += loss_train.get('loss_fbc', 0) 


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
    loss_dict_epoch = {
    "ce": loss_ce / n_iter,
    "loss_con": loss_con / n_iter,
    "loss_supcon": loss_supcon / n_iter,
    "loss_ucon": loss_ucon / n_iter,
    "loss_src_ucon": loss_src_ucon / n_iter,
    "loss_src_supcon": loss_src_supcon / n_iter,
    "loss_tgt_ucon": loss_tgt_ucon / n_iter,
    "loss_tgt_supcon": loss_tgt_supcon / n_iter,
    "loss_fbc": loss_fbc / n_iter
    }        
    print(loss_dict_epoch)
    return log_wandb,model,1,total_step,loss_dict_epoch

def val_epoch(
    mode,dataset_name,val_loader,device,log_wandb,model,total_step,lr,epoch,config,metric_collection
):
    model.eval()
    epoch_loss=0

    batch_iter = 0
    n_iter=len(val_loader)
    predict_val_list=[]
    label_list=[]
    
    for i ,(x,y,_,_,_) in enumerate(val_loader):

        batch_iter=batch_iter+config.batch_size
        total_step+=1
        if config.method!='WiGRUNT' and config.method!='WiSDA' and config.method!='SourceOnly2': 
            x = x.permute(0, 1, 2, 4, 3)

        x=x.float().to(device)
        y=y.to(device) 

        predict_val,loss_val=model.predict(x,y)
        epoch_loss+=loss_val
        predict_val_list.append(predict_val)
        label_list.append(y)
        
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
    
    print(f"Epoch {epoch} validation completed. Average loss: {epoch_loss:.4f}, Accuracy: {epoch_metrics['accuracy']:.4f}")
    acc_manual = (torch.cat(predict_val_list, dim=0).argmax(dim=-1) == torch.cat(label_list, dim=0)).float().mean().item()
    print(f"Epoch {epoch} manual accuracy: {acc_manual:.4f}")


    
    return log_wandb,model,1,total_step,epoch_metrics['accuracy'],epoch_loss
    
def train(config,log_wandb,optuna_trial=None):

    device = torch.device(config.device)
    
    ###########################读取数据集###################################
    if config.csidataset=='CSIDA':
        train_dataset, val_dataset, test_dataset = get_dataset(config,None)
    elif config.csidataset=='Widar3.0':
        train_dataset, val_dataset, test_dataset = get_dataset(config,None)
    elif config.csidataset=='SignFi':
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
    
    
    ###########################得到backbone、以及对应算法###################################
    if config.backbone=='CSIResNet':
        backbone=Resnet_enc(config.inputshape,None) 
    elif config.backbone=='ResNet':
        backbone=None
    else:
        raise ValueError(f"{config.backbone} is not supported")
    T_max=100 * max(len(train_loader),len(test_loader))
    if config.method=='SourceOnly':
        model=contrast_methods.SourceOnly(config,backbone,T_max) #算法定义成功
    elif config.method=='UniCrossFi_uda_hardpseudo_fbc':
        model=contrast_methods.UniCrossFi_uda_hardpseudo_fbc(config,backbone,T_max)
    elif config.method=='AdvSKM':
        model=contrast_methods.AdvSKM(config,backbone,T_max)
    elif config.method=='CoTMix':
        model=contrast_methods.CoTMix(config,backbone,T_max)
    elif config.method=='FewSense':
        model=contrast_methods.FewSense(config,backbone,T_max)
    elif config.method=='WiGRUNT':
        model=contrast_methods.WiGRUNT(config,backbone,T_max)
    elif config.method=='WiSDA':
        model=contrast_methods.WiSDA(config,backbone,T_max)
    else:
        raise ValueError("Invalid method setting.") #算法定义成功
    model=model.to(device=device)
    ###########################得到模型###################################
    

    ###########################保存路径+各种utils变量###################################
    check_point_path=f'{config.output_dir}/{config.method}/{config.csidataset}/{config.cross_domain_type}/{config.target_domain}/'
    os.makedirs(check_point_path, exist_ok=True)
    total_step = 0  # logging step
    total_stop_v=0
    non_improve_epoch=0
    best_val_acc=0.0
    best_epoch=0
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
    train_loss_history = []
    val_loss_history = []
    for epoch in range(config.epochs):
        
        #训练过程
        # print(epoch)
        # if config.csidataset=='CSIDA' and epoch<config.pseudo_start_epoch:
        #     non_improve_epoch = 0 
        if config.method=='UniCrossFi_uda_hardpseudo_fbc' and epoch>=config.pseudo_start_epoch:
                model.compute_prototypes(train_loader,test_loader)
        log_wandb,model,lr,total_step,loss_dict=train_epoch(mode='train',dataset_name=config.csidataset,train_loader=train_loader,model=model
                                                  ,test_loader=test_loader,device=device,log_wandb=log_wandb,
                                                                    total_step=total_step,lr=lr,epoch=epoch,config=config,metric_collection=metric_collection)
        train_loss_history.append(loss_dict) 
        with torch.no_grad():
            log_wandb,model,lr,total_step_v,this_acc,val_epoch_loss=val_epoch(mode='val',dataset_name=config.csidataset,val_loader=val_loader,model=model,
                                                    device=device,log_wandb=log_wandb,
                                                                        total_step=total_stop_v,lr=lr,epoch=epoch,config=config,metric_collection=metric_collection)
            val_loss_history.append(val_epoch_loss)  
            if this_acc>best_val_acc:
                non_improve_epoch=0
                best_val_acc=this_acc
                best_epoch=epoch
                torch.save(model.state_dict(), f'{check_point_path}/checkpoint_seed{config.seed}.pth')
                # print(f'{epoch}数据已经保存')
            else:
                non_improve_epoch=non_improve_epoch+1
            if non_improve_epoch > config.early_stop_epoch:
                break
            
    model.load_state_dict(torch.load(f'{check_point_path}/checkpoint_seed{config.seed}.pth', map_location=device))
    result_file = os.path.join(check_point_path, "result_multiseed.txt")
    model.eval()
    predict_test_list=[]
    label_list=[]
    if config.method=='FewSense':
        model.compute_prototypes(train_loader)
    with torch.no_grad():
        # tbar=tqdm(test_loader)
        for i ,(x,y,_,_,_) in enumerate(test_loader):
            
            if config.method!='WiGRUNT' and config.method!='WiSDA' and config.method!='SourceOnly2': #只有CSI格式需要转换
                x = x.permute(0, 1, 2, 4, 3)
            
            x=x.float().to(device)
            y=y.to(device)
            
            predict_test,loss_test=model.predict(x,y)
            predict_test_list.append(predict_test)
            label_list.append(y)
            batch_metrics=metric_collection.forward(predict_test.softmax(dim=-1),y.int())
            # print(batch_metrics)
        epoch_metrics = metric_collection.compute()
        for k in epoch_metrics.keys():
            log_wandb.log({f'epoch_test_{str(k)}': epoch_metrics[k],
                        'epoch': 0})  # log epoch metric 
       
        metric_collection.reset()
        acc_manual = (torch.cat(predict_test_list, dim=0).argmax(dim=-1) == torch.cat(label_list, dim=0)).float().mean().item()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        # -------- 绘制 loss 曲线 --------
        plt.figure(figsize=(8,6))
        epochs = range(1, len(train_loss_history)+1)

        # 画 train loss 各个元素
        for key in train_loss_history[0].keys():
            plt.plot(epochs, [d[key].detach().cpu().item() if torch.is_tensor(d[key]) else d[key] 
                            for d in train_loss_history], 
                    label=f"train_{key}")

        # 画 val loss
        val_loss_plot = [v.detach().cpu().item() if torch.is_tensor(v) else v for v in val_loss_history]
        plt.plot(epochs, val_loss_plot, label="val_loss", linestyle="--", color="black")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(check_point_path, "loss_curve.png"))
        plt.close()



# 打开文件追加写入
        with open(result_file, "a") as f:
            # 写入实验环境信息
            f.write(f"--- Experiment Environment ---\n")
            f.write(f"Time: {now}\n")
            f.write(f"Learning Rate (config.lr): {config.lr}\n")
            f.write(f"Weight Decay (config.weight_decay): {config.weight_decay}\n")
            f.write(f"Seed (config.seed): {config.seed}\n")
            


            # 写入 best_val_acc 信息
            f.write(f"\nBest Validation Accuracy at {int(best_epoch)} (best_val_acc): {best_val_acc:.4f}  # 训练过程中验证集上出现的最高准确率\n")

            # 写入 acc_manual 和 epoch_metrics['accuracy'] 的信息
            f.write(f"Test Manual Accuracy (acc_manual): {acc_manual:.4f}  # 自定义手动计算得到的准确率\n")
            f.write(f"Logged Accuracy (epoch_metrics['accuracy']): {epoch_metrics['accuracy'].item():.4f}  # 自动日志记录的准确率指标\n")
            f.write(f"ratio (config.ratio): {config.ratio}\n")
            # 写入所有 epoch_metrics 中的内容
            f.write(f"\n--- Epoch Metrics ---\n")
            for k in epoch_metrics.keys():
                line = f"{k}: {epoch_metrics[k].item():.4f}\n"
                f.write(line)

            # 末尾添加分隔符
            f.write("\n" + "=" * 40 + "\n\n")
        log_wandb.log(
            {'test_accuracy_manual':acc_manual,}
        )
        print(f"Record test manual accuracy: \n{100*acc_manual:.2f}") 

   
    
    
 
