import argparse

import sys
import os

# 获取当前文件的上一级目录
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)
from data import  dataset
from models import contrast_methods
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import manifold, datasets








parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,default="/opt/data/private/ablation_study/data_widar_800/tdcsi/")#"/mnt/ssd1/LiuSJ/") #数据的地址 
# parser.add_argument('--data_dir', type=str,default="/opt/data/private/ablation_study/data_widar_wigrunt")#"/mnt/ssd1/LiuSJ/") #数据的地址 

parser.add_argument('--csidataset', type=str, default='Widar3.0')#'Widar3'#'CSIDA',#'SignFi'  数据集有哪些  
parser.add_argument('--backbone', type=str, default="ResNet") #使用的 backbone
parser.add_argument('--project_name', type=str, default="unknow_method") #project_name
parser.add_argument("--freeze_enc", action="store_true", 
                    help="是否冻结 feature_extractor不训练 encoder")



# CSIDA是 [684,1800] Widar3.0 [180,2500]/[120,2500] SignFi [180,500]
#网络参数相关
parser.add_argument('--method',default='ERM') #使用什么跨域方法
parser.add_argument('--inputshape', type=int, nargs=2, default=[60, 800]) #228, 1800  #60,800(widar)
parser.add_argument('--classify',type=str,default='linear') # 分类头线性or非线性 
parser.add_argument('--last_dim',type=int,default=512) #backbone的形状 
parser.add_argument('--num_classes',type=int,default=6)

#zzy加的 任务相关
parser.add_argument("--task_setting", default="UDA", help="various experiment setting")
parser.add_argument("--cross_domain_type", default="orientation", help="set cross domain type")
parser.add_argument("--early_stop_epoch", type=int,default=20, help="contral early stop")
parser.add_argument("--pseudo_start_epoch", type=int,default=30, help="from which epoch to use pseudo label")

#wandb相关
# parser.add_argument("--project_name", default="FDAARC_UDA", help="set wandb project name")
parser.add_argument("--run_name", default="Sourceonly", help="set run_name for wandb")

#优化器相关 
parser.add_argument("--optimizer", default="Adam", help="set optimizer for our project")
parser.add_argument("--momentum", default=0.9, help="momentum for sgd")
parser.add_argument("--weight_decay", default=5e-4,type=float, help="set weight decay for sgd")





#数据集相关    
parser.add_argument('--data_type', type=str, default="amp+pha") #一般都是 amp+pha 这个目前没用 
parser.add_argument('--source_domain', type=str, default=None)
parser.add_argument('--target_domain', type=str, default='orientation2')
# parser.add_argument("--FDA_hp", default=[(10,10),(10,50),(10,20),(10,30),(10,40),(10,60),(10,70),(10,80),(10,100),(10,200),(10,300),(10,400),(10,1000)], type=ast.literal_eval, help="test bandwith in FDA")

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--results_file', type=str, default="test_results_uda.txt")
parser.add_argument('--checkpoint_freq', type=int, default=5 ) # 每几轮保存
parser.add_argument('--checkpoint_path', type=str, default="/opt/data/private/FDAARC/checkpoints/ERM/Widar3.0/orientation/orientation2/checkpoint_seed42.pth" ) # 从哪里加载模型
parser.add_argument('--output_dir', type=str, default='/opt/data/private/FDAARC/checkpoints') # 模型保存到哪里
parser.add_argument('--a_select', type=str, default='4_5')
parser.add_argument('--ratio', type=float, default=None)



# 算法相关 
# SimCLR
parser.add_argument('--noise_std', type=float, default=0.01)
# UniCrossFi
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--pseudo_label_threshold", type=float, default=0.75)
parser.add_argument("--temperature", type=float, default=0.07)
parser.add_argument('--scale_cap',type=int,default=10)
# WiSR
parser.add_argument('--w_adv', type=float, default=0.1) 
#AdkSKM
parser.add_argument('--domain_loss_weight', type=float, default=1) #AdvSKM相关参数
parser.add_argument('--src_loss_weight', type=float, default=1) #AdvSKM相关参数
#CoTMix
parser.add_argument('--src_cls_weight', type=float, default=1e-3) #CoTMix相关参数
parser.add_argument('--temporal_shift', type=int, default=5) #CoTMix相关参数
parser.add_argument('--mix_ratio', type=float, default=0.9) #CoTMix相关参数
parser.add_argument('--src_supCon_weight', type=float, default=1e-3) #CoTMixM相关参数
parser.add_argument('--trg_cont_weight', type=float, default=1e-3) #CoTMix相关参数
parser.add_argument('--trg_entropy_weight', type=float, default=1e-3) #CoTMix相关参数

parser.add_argument('--wisda_loss_weight', type=float, default=1e-3) #WiSDA相关参数

parser.add_argument('--wiopen_temperature', default=0.05, type=float,
                    help='temperature parameter for softmax')
parser.add_argument('--wiopen_memory_momentum', default=0.5, type=float,
                        help='momentum for non-parametric updates')

parser.add_argument("--mix_alpha", default=0.7, type=float) #LiSA参数 

parser.add_argument("--interval_signfi", default=80, type=int) #SignFi参数  这个参数是5-250 最好是5的倍数
parser.add_argument("--CSIDA_freq", default=400, type=int) #SignFi参数  这个参数是5-250 最好是5的倍数
config = parser.parse_args()
train_dataset, val_dataset, test_dataset = dataset.get_dataset(config,None)
if config.method == 'WiSR':
    config.num_domains = len(train_dataset.get_all_domain_labels())  # 获取源域数据集的域标签数量

def get_data(config):
    # train_dataset, val_dataset, test_dataset = dataset.get_dataset(config,None)
    
    # print(len(train_dataset),len(val_dataset),len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=15) #得到所有的loader
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size , shuffle=False,num_workers=15)
    test_loader=DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False,num_workers=15)
    return train_loader,val_loader,test_loader

def get_model(config):
    if config.method == 'UniCrossFi_dg':
        model= contrast_methods.UniCrossFi_dg(config,None,100)
    elif config.method == 'UniCrossFi_semidg':
        model= contrast_methods.UniCrossFi_semidg(config,None,100)
    elif config.method == "WiSR":
        model = contrast_methods.WiSR(config, None, 100)
    elif config.method == 'WiGRUNT':
        model = contrast_methods.WiGRUNT(config, None, 100)
    elif config.method == 'ERM':
        model = contrast_methods.ERM(config, None, 100)
    # print(model)
    return model

import torch
def test_acc(loader,model):
    model=model.to(config.device)
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)

    model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    device = config.device
    model.eval()
    predict_test_list=[]
    label_list=[]
    emb_list = []
    with torch.no_grad():
        for i ,(x,y,_,_,_) in enumerate(loader): #DG
        # for i ,(x,y) in enumerate(loader): #UDA


            if config.method!='WiGRUNT' and config.method!='WiSDA' and config.method!='SourceOnly2': #只有CSI格式需要转换
                x = x.permute(0, 1, 2, 4, 3)

            x=x.float().to(device)
            y=y.to(device)
            
            print(x.shape)

            _=model.predict4time(x,y)


            predict_test,loss_test=model.predict(x,y)
            predict_test_list.append(predict_test.cpu())
            
            if config.method == 'UniCrossFi_dg' or config.method == 'UniCrossFi_semidg':    
                emb,b,n_views,dim = model.get_emb(x,y)
                if config.method =='UniCrossFi_dg':
                    emb = emb.view(b,n_views,-1)
                emb_mean = emb.mean(dim=1)
            elif config.method == 'WiGRUNT':
                emb_mean = model.get_feature(x)
            else:
                emb_mean = model.get_emb(x,y)
                # print(emb_mean.shape)
            # b,n_views,dim = emb.shape

            # print(emb_mean.shape)
            # exit(0)
            emb_list.append(emb_mean.cpu())
            label_list.append(y.cpu())
    emb_all = torch.cat(emb_list, dim=0)      # [N, dim]
    label_all = torch.cat(label_list, dim=0)  # [N]
    emb_all_np = emb_all.numpy()
    label_all_np = label_all.numpy()
    acc_manual = (torch.cat(predict_test_list, dim=0).argmax(dim=-1) == torch.cat(label_list, dim=0)).float().mean().item()
    print(acc_manual)
    return emb_all_np,label_all_np

    # print(emb_all_np.shape,label_all_np.shape)

    # print("我到了")


dict_widar = {
    "Push&Pull": 0,
    "Sweep": 1,
    "Clap": 2,
    "Slide": 3,
    "DrawO(Horizontal)": 4,
    "DrawZigzag(Horizontal)": 5
}
class t_sne():
    def __init__(self,n_components,perplexity,early_exaggeration,learning_rate,n_iter):
        super(t_sne, self).__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.tsne = manifold.TSNE(n_components=n_components,perplexity=perplexity,early_exaggeration=early_exaggeration,
                                  learning_rate=learning_rate,n_iter=n_iter,init='pca')

    def run_t_sne(self,X):
        X_rteshape = X.reshape(X.shape[0], -1)
        self.n_samples, self.n_features = X_rteshape.shape
        self.X_tsne = self.tsne.fit_transform(X_rteshape)
    def visulization(self, y, config):
        y = np.squeeze(y)
        x_min, x_max = self.X_tsne.min(0), self.X_tsne.max(0)
        X_norm = (self.X_tsne - x_min) / (x_max - x_min)  # 归一化

        plt.figure(figsize=(12, 12))
        colors = plt.cm.Set1(np.arange(len(dict_widar)))  # 6种颜色

        # 按类别画图，并加legend
        for action, label in dict_widar.items():
            idx = y == label
            plt.scatter(
                X_norm[idx, 0], X_norm[idx, 1],
                c=[colors[label]],
                label=action,
                marker='*',
                s=50,
                alpha=0.8
            )

        plt.xticks([])
        plt.yticks([])
        plt.legend(loc='best', fontsize=10)
        save_path = f"/opt/data/private/FDAARC/util/{config.method}_0.7.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 可视化结果已保存到: {os.path.abspath(save_path)}")

    def run_2_t_sne(self, X1, X2):
        X1_rteshape = X1.reshape(X1.shape[0], -1)
        X2_rteshape = X2.reshape(X2.shape[0], -1)
        X_rteshape = np.concatenate((X1_rteshape,X2_rteshape))
        self.n_samples, self.n_features = X_rteshape.shape
        self.X_tsne = self.tsne.fit_transform(X_rteshape)

    def visulization2(self,y1,y2):
        y1 = np.squeeze(y1)
        y2 = np.squeeze(y2)
        y = np.concatenate((y1,y2))
        x_min, x_max = self.X_tsne.min(0), self.X_tsne.max(0)
        X_norm = (self.X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(12, 12))
        for i in range(y1.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], "*", color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        for i in range(y1.shape[0],y2.shape[0]+y1.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], ".", color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()



def test_time(loader,model):
    model=model.to(config.device)
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)

    model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    device = config.device
    model.eval()
    predict_test_list=[]
    label_list=[]
    emb_list = []
    start_time=time.time()
    with torch.no_grad():
        for i ,(x,y,_,_,_) in enumerate(loader): #DG
        # for i ,(x,y) in enumerate(loader): #UDA
   
            x = x.permute(0, 1, 2, 4, 3)

            x=x.float().to(device)
            y=y.to(device)

            _=model.predict4time(x,y)

    end_time=time.time()
    print(f"start_time {start_time} end time {end_time} all_time {end_time-start_time}")
    exit(0)


_,_,test_loader = get_data(config)
model=get_model(config)
test_time(test_loader,model)
emb,label = test_acc(test_loader,model)
t = t_sne(n_components=2,perplexity=50,early_exaggeration=6,learning_rate=300,n_iter=1000)
# t = t_sne(n_components=2,perplexity=50,early_exaggeration=6,learning_rate=300,n_iter=1000)
t.run_t_sne(emb)
t.visulization(label,config)

