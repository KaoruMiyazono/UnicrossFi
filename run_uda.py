# run_uda``.py
import os
from collections import Counter
import argparse
import random
import sys
import re
import ast
import datetime
from pathlib import Path
# from FDAARC.models import backbones, heads
import numpy as np
import PIL
import torch
import torchvision
import wandb

from util.logger import Logger
from util import hparams_registry
from algorithm import train_uda,train_uda_wiopen,train_dg,train_uda_unicrossfi_fdb

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_TENSORRT_WARNING'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# exit(0)

def main(inargs=None,wandb_run=None,optuna_trial=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default="/opt/data/data_widar_800/tdcsi/") #dataset position 
    parser.add_argument('--csidataset', type=str, default='Widar3.0')#'Widar3'#'CSIDA'  
    parser.add_argument('--backbone', type=str, default="ResNet") 
    parser.add_argument('--project_name', type=str, default="unknow_method") #project_name
    parser.add_argument("--freeze_enc", action="store_true", 
                        help="whether to freeze encoder during training")

    # CSIDA是 [684,1800] Widar3.0 [180,2500]/[120,2500]
    #network parameters
    parser.add_argument('--method',default='UniCrossFi_dg') 
    parser.add_argument('--inputshape', type=int, nargs=2, default=[60, 800]) #228, 1800 (CSIDA) #60,800(widar)
    parser.add_argument('--classify',type=str,default='linear') # linear classifier or not 
    parser.add_argument('--last_dim',type=int,default=512)  
    parser.add_argument('--num_classes',type=int,default=6)
    
    #task realted
    parser.add_argument("--task_setting", default="UDA", help="various experiment setting")
    parser.add_argument("--cross_domain_type", default="location", help="set cross domain type")
    parser.add_argument("--early_stop_epoch", type=int,default=20, help="contral early stop")
    parser.add_argument("--pseudo_start_epoch", type=int,default=30, help="from which epoch to use pseudo label")
    
    #wandb related
    parser.add_argument("--run_name", default="Sourceonly", help="set run_name for wandb")
    
    #optimizer related 
    parser.add_argument("--optimizer", default="Adam", help="set optimizer for our project")
    parser.add_argument("--momentum", default=0.9, help="momentum for sgd")
    parser.add_argument("--weight_decay", default=5e-4,type=float, help="set weight decay for sgd")

    
    #dataset related    
    parser.add_argument('--data_type', type=str, default="amp+pha") 
    parser.add_argument('--source_domain', type=str, default=None)
    parser.add_argument('--target_domain', type=str, default='location1')
    parser.add_argument("--FDA_hp", default=[(10,10),(10,50),(10,20),(10,30),(10,40),(10,60),(10,70),(10,80),(10,100),(10,200),(10,300),(10,400),(10,1000)], type=ast.literal_eval, help="test bandwith in FDA")

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--results_file', type=str, default="test_results_uda.txt")
    parser.add_argument('--checkpoint_freq', type=int, default=5 ) 
    parser.add_argument('--checkpoint_path', type=str, default=None ) 
    parser.add_argument('--output_dir', type=str, default='/opt/data/checkpoints') 
    parser.add_argument('--a_select', type=str, default='4_5')
    parser.add_argument('--ratio', type=float, default=None)

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
    parser.add_argument('--domain_loss_weight', type=float, default=1) 
    parser.add_argument('--src_loss_weight', type=float, default=1) 
    #CoTMix
    parser.add_argument('--src_cls_weight', type=float, default=1e-3)
    parser.add_argument('--temporal_shift', type=int, default=5) 
    parser.add_argument('--mix_ratio', type=float, default=0.9) 
    parser.add_argument('--src_supCon_weight', type=float, default=1e-3) 
    parser.add_argument('--trg_cont_weight', type=float, default=1e-3) 
    parser.add_argument('--trg_entropy_weight', type=float, default=1e-3)
    # WiSDA
    parser.add_argument('--wisda_loss_weight', type=float, default=1e-3) 
    parser.add_argument('--wiopen_temperature', default=0.05, type=float,
                     help='temperature parameter for softmax')
    parser.add_argument('--wiopen_memory_momentum', default=0.5, type=float,
                         help='momentum for non-parametric updates')

    if inargs is not None and optuna_trial is not None: 
        config = inargs
        log_wandb = wandb_run
        print("📌 情况1：Optuna 调用")
        print("➡️ config (inargs):", config)
    elif wandb_run is not None and inargs is None and optuna_trial is None:
        config = wandb_run.config
        config = argparse.Namespace(**args)
        log_wandb = wandb_run
        print("📌 情况2：WandB Sweep 调用")
        print("➡️ 原始 wandb.config:", config)
    else: 
        args = parser.parse_args()
        now = datetime.datetime.now().strftime("%Y%m%d")
        wandb.login(key="YOURTKEYHERE")  # replace with your wandb key; not necessary if don't use wandb

        log_wandb=wandb.init(
            project=args.project_name,
            config=vars(parser.parse_args()),
            mode="online",
            name=f"{args.method}_{args.target_domain}")
        config = wandb.config
        config=argparse.Namespace(**config)

    best_val_acc=None
    if config.method=='UniCrossFi_uda_hardpseudo_fbc':
        best_val_acc = train_uda_unicrossfi_fdb.train(config,log_wandb,optuna_trial)
    elif (config.method=="WiSR" or 
    config.method=="ERM" or 
    config.method=="SimCLR"):
        best_val_acc=train_dg.train(config,log_wandb,optuna_trial)
    elif config.method=='UniCrossFi_dg':
        best_val_acc=train_dg.train(config,log_wandb,optuna_trial)
    elif config.method!='Wiopen':
        train_uda.train(config,log_wandb)
    else:
        train_uda_wiopen.train(config,log_wandb)
    
    if inargs is None and wandb_run is None and optuna_trial is None:
        log_wandb.finish() 

    if best_val_acc is not None:
        return best_val_acc


def load_model(model_class, path, device):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)

if __name__ == '__main__':
    main()