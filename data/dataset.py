import torch
import numpy as np
from .CSIDA import load_csida_data_uda,load_csida_data_OSFSUDA,load_csida_data_UDA_WiSDA,load_csida_data_UDA_WiGRUNT,load_csida_data_UDA_Wiopen,load_csida_data_DG
import random
from .Widar import load_widar_data_DG_DARC,load_widar_data_uda,load_widar_data_OSFSUDA,load_widar_data_UDA_WiGRUNT,load_widar_data_UDA_Wiopen,load_widar_data_uda_new,load_widar_data_UDA_2d2a,load_widar_data_UDA_FAGes2d2a,visual_dataset,load_widar_data_UDA_visual,load_widar_data_UDA_visual_1d,load_widar_data_uda_DG,load_widar_data_uda_DG_final


def set_seed(seed: int):
    random.seed(seed)                          # Python 随机模块
    np.random.seed(seed)                       # NumPy 随机模块
    torch.manual_seed(seed)                    # PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)               # PyTorch GPU 随机种子（单卡）
    torch.cuda.manual_seed_all(seed)           # 多卡情况下的随机种子

    # 确保 cudnn 的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(args, hparams):

    set_seed(0)
    if args.csidataset=='CSIDA':
        if args.task_setting=="UDA":
            if args.method=="WiSDA":
                # pass
                train_dataset, val_dataset, test_dataset=load_csida_data_UDA_WiSDA(args,hparams)
            elif args.method=='WiGRUNT': 
                train_dataset,val_dataset,test_dataset=load_csida_data_UDA_WiGRUNT(args,hparams)
            elif args.method=='Wiopen':
                train_dataset,val_dataset,test_dataset=load_csida_data_UDA_Wiopen(args,hparams)
            elif args.method=="AdvSKM" or args.method=='CoTMix':
                train_dataset,val_dataset,test_dataset=load_csida_data_uda(args,hparams)
            else:  #lisa,unicrossfi,erm
                train_dataset, val_dataset, test_dataset = load_csida_data_DG(args, hparams)
            return train_dataset, val_dataset, test_dataset
        else :
            raise ValueError("Invalid task setting. Choose 'UDA'.")

    elif args.csidataset=='Widar3.0':
        if args.task_setting=="UDA":
            if args.method=='WiGRUNT' or args.method=='WiSDA' or args.method=='SourceOnly2':
                train_dataset,val_dataset,test_dataset=load_widar_data_UDA_WiGRUNT(args, hparams)
            elif args.method=='Wiopen':
                train_dataset,val_dataset,test_dataset=load_widar_data_UDA_Wiopen(args, hparams)
            elif args.method=='visual':
                train_dataset, val_dataset, test_dataset = load_widar_data_UDA_visual(args,hparams)
            elif args.method=='visual_1d':
                train_dataset, val_dataset, test_dataset = load_widar_data_UDA_visual_1d(args,hparams)
            elif (args.method == 'UniCrossFi_dg' or
                args.method == 'UniCrossFi_semidg' or
                args.method == 'DG' or
                args.method == 'ERM' or
                args.method == 'SimCLR' or
                args.method == 'WiSR' or
                args.method == 'UniCrossFi_uda_hardpseudo_fbc'):
                train_dataset, val_dataset, test_dataset = load_widar_data_uda_DG_final(args, hparams)
            else :
                # 这里相对之前改动的点是对于Widar 整体把他读到内存里问题很大 不如边读路径边读
                if 'wiopen' in args.data_dir or '00' in args.data_dir: # source only进入这里
                    train_dataset, val_dataset, test_dataset = load_widar_data_UDA_FAGes2d2a(args, hparams)
                else:
                    train_dataset, val_dataset, test_dataset = load_widar_data_uda(args, hparams)
            return train_dataset, val_dataset, test_dataset
        else:
            raise ValueError("Invalid task setting. Choose 'UDA'.")
    else:
        raise ValueError(f"{args.csidataset} are not supported")
        
        
# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item() #得到使用的label
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations

