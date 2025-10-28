# 优化器 
import torch
#传入的是个字典
def get_optimizer(params,config):
    optumizer = config.optimizer #得到优化器 
    
    if optumizer == 'SGD':
        optimizer = torch.optim.SGD(
                params,
                lr=config.lr,
                momentum=0.9,
                weight_decay=config.weight_decay,
                dampening=0,
                nesterov=False,
            )
    elif optumizer == 'Adam':
        optimizer = torch.optim.Adam(
                params,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
    else:
        raise ValueError(f"Unsupported optimizer: {optumizer}")    
    return optimizer
    