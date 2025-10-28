import optuna
import wandb
import argparse
import ast
from run_uda import main
import os
os.environ["WANDB_CONSOLE"] = "off"

def objective(trial):
    # === Searching Space ===
    lr = trial.suggest_float('lr', 1e-5, 5e-3)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-3)
    tau = trial.suggest_float('temperature', 0.07, 0.2)
    scale_cap = trial.suggest_categorical("scale_cap", [4, 6, 10])
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight_decay", type=float, default=wd)
    parser.add_argument("--temperature", type=float, default=tau)
    parser.add_argument('--scale_cap',type=int,default=scale_cap)
    ###############################
    ## Basic running and output configuration
    parser.add_argument('--target_domain', type=str, default='room2')
    parser.add_argument("--cross_domain_type", default="room", help="set cross domain type")

    parser.add_argument('--data_dir', type=str,default="/opt/data/data_widar_800/tdcsi/")
    parser.add_argument('--csidataset', type=str, default='Widar3.0')#'Widar3.0'#'CSIDA'
    parser.add_argument('--backbone', type=str, default="ResNet") 
    #network parameters
    parser.add_argument('--method',default='UniCrossFi_dg')
    parser.add_argument('--inputshape', type=int, nargs=2, default=[60, 800]) #228, 1800 (CSIDA) #60,800(widar)
    parser.add_argument('--classify',type=str,default='linear') # classifier linear or not 
    parser.add_argument('--last_dim',type=int,default=512) #last_dim of backbone 
    parser.add_argument('--num_classes',type=int,default=6)

    #task realted
    parser.add_argument("--task_setting", default="UDA", help="various experiment setting")
    parser.add_argument("--early_stop_epoch", type=int,default=20, help="contral early stop")
    #wandb related
    parser.add_argument("--run_name", default="UniCrossFi", help="set run_name for wandb")
    parser.add_argument('--project_name', type=str, default="Unicrossfi") #project_name for wandb
    #optimizer related 
    parser.add_argument("--optimizer", default="Adam", help="set optimizer for our project")
    parser.add_argument("--momentum", default=0.9, help="momentum for sgd")

    #dataset related    
    parser.add_argument('--data_type', type=str, default="amp+pha") 
    parser.add_argument('--source_domain', type=str, default=None)
    parser.add_argument("--FDA_hp", default=[(10,10),(10,50),(10,20),(10,30),(10,40),(10,60),(10,70),(10,80),(10,100),(10,200),(10,300),(10,400),(10,1000)], type=ast.literal_eval, help="test bandwith in FDA")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--results_file', type=str, default="test_results_unicrossfi.txt")
    parser.add_argument('--checkpoint_freq', type=int, default=5 ) # save per n epochs
    parser.add_argument('--checkpoint_path', type=str, default=None ) # import model from here
    parser.add_argument('--output_dir', type=str, default='./checkpoints') # save model to here
    parser.add_argument('--a_select', type=str, default='4_5')
    parser.add_argument('--ratio', type=float, default=None)

    # SimCLR
    parser.add_argument('--noise_std', type=float, default=0.01)
    # UniCrossFi
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--pseudo_label_threshold", type=float, default=0.98)
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


    args = parser.parse_args()
    args.output_dir = f"/opt/data/checkpoints/UniCrossFi/DG_T={args.target_domain}_trial{trial.number}"
    if args.csidataset=='CSIDA':
        args.inputshape=[228, 1800]
    else:
        args.inputshape=[60, 800] 
    wandb.login(key="YOURTKEYHERE")  # replace with your wandb key; not necessary if don't use wandb
    run = wandb.init(
        project="Unicrossfi",
        name=f"trial_{args.target_domain}_{trial.number}",
        mode="offline",
        config={
            "lr": lr,
            "weight_decay": wd,
            "temperature": tau,
            "scale_cap": scale_cap,
            "target_domain": args.target_domain,
            "csidataset": args.csidataset
            },
        reinit=True,
    )
    main(inargs=args,wandb_run=run,optuna_trial=trial) 
    run.finish()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, n_jobs=1) 
    print("Best params:", study.best_trial.params)
