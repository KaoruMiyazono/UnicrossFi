from data.utils import DARCDataset,BaseCSIDataset,select_data_with_class,select_data_with_k_shot,WidarDataset,WiGRUNT_dataset,Wiopen_dataset,d2a2dataset,FAGes_d2a2dataset,visual_dataset,visual_dataset_1d,DGDataset,LiSADataset
import os 
import numpy as np 
from torch.utils.data import ConcatDataset, random_split
from sklearn.model_selection import train_test_split


#TODO  先去统计一下Widar3.0
#  每个数据以下格式 20181128_Clap_user_user6_location_4_orientation_2_repetition_2.npy

#widar的字典 
dict_widar = {
    "Push&Pull": 0,
    "Sweep": 1,
    "Clap": 2,
    "Slide": 3,
    "DrawO(Horizontal)": 4,
    "DrawZigzag(Horizontal)": 5
}

def get_Widar_domain(target_domain,cross_domain_type,root_dir):
    widar_all_room=['room1','room2','room3']
    widar_all_user=['user1','user2','user3','user4','user5','user6','user7','user8','user9','user10','user11','user12','user13','user14','user15','user16','user17']
    widar_all_location=['location1','location2','location3','location4','location5','location6','location7','location8']
    widar_all_orientation = ['orientation1', 'orientation2', 'orientation3', 'orientation4', 'orientation5']

    if cross_domain_type=='room':
        # pass
        # 这里需要
        target_domian=[target_domain]
        source_domain=[source for source in widar_all_room if source not in target_domian]

        
    elif cross_domain_type=='user':
        # pass
        target_domain=[target_domain]
        source_domain=[source for source in widar_all_user if source not in target_domain]

    elif cross_domain_type=='location':
        target_domain=[target_domain]
        source_domain=[source for source in widar_all_location if source not in target_domain]

    elif cross_domain_type=='orientation':
        target_domain=[target_domain]
        source_domain=[source for source in widar_all_orientation if source not in target_domain]

        # pass
    else:
        raise ValueError(f"Invalid cross_domain_type: {cross_domain_type}")
    print(f"source_domain如下 {source_domain}")
    print(f"target_domain如下 {target_domain}")
    # exiy(0)
    # print(target_domain)
    return source_domain,target_domain

def get_widar_data(source_domain,target_domain,root_dir,cross_domain_type):
    source_data=[]
    source_label=[]
    target_data=[]
    targer_label=[]
    cnt=0
    print("🚀 正在加载数据...")
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                # "20181109_room_room1_Clap_user_user1_location_1_orientation_1_repetition_5.npy"
                file_path = os.path.join(root, file)
                
                data = np.load(file_path) #得到 数据 
                # print(file)
                # print(data.shape)
                parts = file.split('_')
                parsed = {
                    'date': parts[0],
                    'room': parts[2],
                    'gesture': parts[3],
                    'user': parts[5],
                    'location': 'location'+parts[7],
                    'orientation': parts[9],
                    'repetition': parts[11]
                }
                label=dict_widar[parsed['gesture']] #得到标签 
                domain_this_file=parsed[cross_domain_type] #得到域信息 
                if domain_this_file in source_domain:
                    source_data.append(data)
                    source_label.append(label)
                elif domain_this_file in target_domain:
                    target_data.append(data)
                    targer_label.append(label)
                else:
                    raise ValueError(f"Invalid domain: {domain_this_file}")
                cnt=cnt+1
                if(data.shape[0]==0):
                    print(f"数据为空 {file_path}")
                    exit(0)
                # print(f"cnt={cnt}  {file_path}  {domain_this_file}  {label} {data.shape}")
    print(type(source_data))
    print(len(source_data))
    print(all(x.shape == source_data[0].shape for x in source_data))
    print(all(x.shape == target_data[0].shape for x in target_data))
    # exit(0)
    source_data_np = np.array(source_data)
    source_label_np = np.array(source_label)
    target_data_np = np.array(target_data)
    targer_label_np = np.array(targer_label)
    
    print(source_data_np.shape)
    print(target_data_np.shape)
    source_data_np=source_data_np.transpose(0, 3, 2, 1, 4)
    target_data_np=target_data_np.transpose(0, 3, 2, 1, 4)
    
    
                
                
                
                

    
    print("🚀 数据加载完成！")
    print(f"source domain样本形状 {source_data_np.shape}")
    print(f"source domain标签形状 {source_label_np.shape}")
    print(f"target domain样本形状 {target_data_np.shape}")
    print(f"target domain标签形状 {targer_label_np.shape}")
    
    return source_data_np, source_label_np, target_data_np, targer_label_np

    

import os

def get_widar_filelists_quick(source_domain, target_domain, root_dir, cross_domain_type):
    source_data_paths = []
    source_labels = []
    target_data_paths = []
    target_labels = []

    cnt = 0
    cnt_no=0
    print("🚀 正在收集文件路径...")
    # print(root_dir)
    # exit(0)

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy") or file.endswith(".png") or file.endswith(".jpg")  or file.endswith(".mat"):
                file_path = os.path.join(root, file)
                # print(file_path)
                
                parts = file.split('_')
                parsed = {
                    'date': parts[0],
                    'room': parts[2],
                    'gesture': parts[3],
                    'user': parts[5],
                    'location': 'location'+parts[7],
                    'orientation': 'orientation' + parts[9],
                    'repetition': parts[11]
                }
                # print(parsed)
                # exit(0)
                label = dict_widar[parsed['gesture']]
                domain_this_file = parsed[cross_domain_type]

                if domain_this_file in source_domain:
                    source_data_paths.append(file_path)
                    source_labels.append(label)
                elif domain_this_file in target_domain:
                    target_data_paths.append(file_path)
                    target_labels.append(label)
                else:
                    cnt_no=cnt_no+1
                    # raise ValueError(f"Invalid domain: {domain_this_file}")

                cnt += 1

    print("🚀 文件路径收集完成！")
    print(f"共计采集到 {cnt} 个文件")
    print(f"源域数量：{len(source_data_paths)}，目标域数量：{len(target_data_paths)}")

    return source_data_paths, source_labels, target_data_paths, target_labels

    

 


from torch.utils.data import random_split

def load_widar_data_uda(args, hparams):
    """
    加载Widar3.0数据集的UDA任务数据,只读取路径和标签
    返回训练集、验证集和测试集,使用自定义WidarDataset
    """
    root_dir = args.data_dir

    # 保持不变：获取源域和目标域的列表
    source_domain, target_domain = get_Widar_domain(args.target_domain, args.cross_domain_type, root_dir)

    # 修改点：只获取路径和标签
    source_paths, source_labels, target_paths, target_labels = get_widar_filelists_quick(
        source_domain, target_domain, root_dir, args.cross_domain_type
    )

    # 使用你待定义的 WidarDataset，输入为路径和标签
    source_dataset = WidarDataset(source_paths, source_labels)
    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])

    test_dataset = WidarDataset(target_paths, target_labels)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    # exit(0)

    return train_dataset, val_dataset, test_dataset


def load_widar_data_UDA_Wiopen(args, hp):
    root_dir = args.data_dir

    # 获取 source 和 target 的域列表
    source_domain_list, target_domain_list = get_Widar_domain(
        args.target_domain,
        args.cross_domain_type,
        root_dir
    )
    print("Source domains:", source_domain_list)
    print("Target domains:", target_domain_list)
    # exit(0)

    # 获取 source 和 target 的路径及标签
    source_img, source_label, target_img, target_label = get_widar_filelists_quick(
        source_domain_list,
        target_domain_list,
        root_dir,
        args.cross_domain_type
    )
    
    
    # exit(0)
    # 构建 Dataset
    from sklearn.model_selection import train_test_split
    train_img, val_img, train_label, val_label = train_test_split(
    source_img, source_label, test_size=0.2,random_state=0
    )

    # 打印结果


    train_dataset=Wiopen_dataset(train_img,train_label)
    val_dataset=Wiopen_dataset(val_img,val_label)
    test_dataset= Wiopen_dataset(target_img,target_label)

    print("📦 训练集样本数:", len(train_dataset))
    print("🧪 验证集样本数:", len(val_dataset))
    print("🧭 测试集样本数:", len(test_dataset))

    return train_dataset, val_dataset, test_dataset

def load_widar_data_UDA_WiGRUNT(args, hp):
    root_dir = args.data_dir

    # 获取 source 和 target 的域列表
    source_domain_list, target_domain_list = get_Widar_domain(
        args.target_domain,
        args.cross_domain_type,
        root_dir
    )
    print("Source domains:", source_domain_list)
    print("Target domains:", target_domain_list)
    # exit(0)

    # 获取 source 和 target 的路径及标签
    source_img, source_label, target_img, target_label = get_widar_filelists_quick(
        source_domain_list,
        target_domain_list,
        root_dir,
        args.cross_domain_type
    )
    # source_dataset = WiGRUNT_dataset(source_img, source_label)
    # train_size = int(0.8 * len(source_dataset))
    # val_size = len(source_dataset) - train_size
    # train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])
    train_img, val_img, train_label, val_label = train_test_split(
    source_img, source_label, test_size=0.2, random_state=0
)

    train_dataset = WiGRUNT_dataset(train_img, train_label)
    val_dataset = WiGRUNT_dataset(val_img, val_label)

    test_dataset = WiGRUNT_dataset(target_img, target_label)

    print("📦 训练集样本数:", len(train_dataset))
    print("🧪 验证集样本数:", len(val_dataset))
    print("🧭 测试集样本数:", len(test_dataset))

    return train_dataset, val_dataset, test_dataset
    
def load_widar_data_UDA_FAGes2d2a(args, hp):
    root_dir = args.data_dir

    # 获取 source 和 target 的域列表
    source_domain_list, target_domain_list = get_Widar_domain(
        args.target_domain,
        args.cross_domain_type,
        root_dir
    )
    print("Source domains:", source_domain_list)
    print("Target domains:", target_domain_list)
    # exit(0)

    # 获取 source 和 target 的路径及标签
    source_img, source_label, target_img, target_label = get_widar_filelists_quick(
        source_domain_list,
        target_domain_list,
        root_dir,
        args.cross_domain_type
    )
    # source_dataset = FAGes_d2a2dataset(source_img, source_label)
    # train_size = int(0.8 * len(source_dataset))
    # val_size = len(source_dataset) - train_size
    # train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])
    train_img, val_img, train_label, val_label = train_test_split(
    source_img, source_label,test_size=0.2, random_state=0
    )

    train_dataset = FAGes_d2a2dataset(train_img, train_label,mode='train',ratio=args.ratio)
    # exit(0)
    
    val_dataset = FAGes_d2a2dataset(val_img, val_label,mode='val')
    test_dataset = FAGes_d2a2dataset(target_img, target_label,mode='test')



    print("📦 训练集样本数:", len(train_dataset))
    print("🧪 验证集样本数:", len(val_dataset))
    print("🧭 测试集样本数:", len(test_dataset))
    # print("我到了")


    return train_dataset, val_dataset, test_dataset
def load_widar_data_UDA_2d2a(args, hp):
    root_dir = args.data_dir

    # 获取 source 和 target 的域列表
    source_domain_list, target_domain_list = get_Widar_domain(
        args.target_domain,
        args.cross_domain_type,
        root_dir
    )
    print("Source domains:", source_domain_list)
    print("Target domains:", target_domain_list)
    # exit(0)

    # 获取 source 和 target 的路径及标签
    source_img, source_label, target_img, target_label = get_widar_filelists_quick(
        source_domain_list,
        target_domain_list,
        root_dir,
        args.cross_domain_type
    )
    source_dataset = d2a2dataset(source_img, source_label)
    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])


    test_dataset = d2a2dataset(target_img, target_label)

    print("📦 训练集样本数:", len(train_dataset))
    print("🧪 验证集样本数:", len(val_dataset))
    print("🧭 测试集样本数:", len(test_dataset))

    return train_dataset, val_dataset, test_dataset

def load_widar_data_OSFSUDA(args, hparams):
    """
    加载Widar3.0数据集的OSFS-UDA任务数据
    """
    # 返回源域训练集、源域验证集、源域测试集、目标域查询集、目标域支持集
    root_dir=args.data_dir
    source_domain,target_domain=get_Widar_domain(args.target_domain,args.cross_domain_type,root_dir) #得到了 domain列表
    source_data,source_label,target_data,target_label=get_widar_data(source_domain,target_domain,root_dir,args.cross_domain_type) #得到数据 
    
    
    ###################################处理source的数据########################################################
    source_data , source_label,_,_,select_set= select_data_with_class(source_data,source_label,"Widar3.0",args.a) #选出来
    train_source_data,test_source_data,train_source_labels,test_source_labels = train_test_split(source_data, source_label, test_size=0.2, random_state=args.seed)
    print(train_source_data.shape)
    print(train_source_labels.shape)
    test_dataset_source=BaseCSIDataset(test_source_data, test_source_labels,transfrom=args.transform) # source的测试集 
    train_source_data,val_source_data,train_source_labels,val_source_labels = train_test_split(train_source_data, train_source_labels, test_size=0.2, random_state=args.seed) # 80% 作为训练集
    
    print(train_source_data.shape)
    print(train_source_labels.shape)
    val_dataset_source=BaseCSIDataset(val_source_data, val_source_labels,transfrom=args.transform) # source的验证集
    train_support, train_label_support, train_query, train_label_query=select_data_with_k_shot(train_source_data,train_source_labels,args.k_train *2 ) # 选择k-shot的支持集和查询集
    print(train_support.shape)
    print(train_label_support)
    train_dataset_source_l=BaseCSIDataset(train_support, train_label_support,transfrom=args.transform)
    train_dataset_source_ul=BaseCSIDataset(train_query, train_label_query,transfrom=args.transform)
    ###################################处理source的数据########################################################
    
    
    
    ###################################处理target的数据########################################################
    
    target_data_in , target_label_in,target_data_not_in,target_label_not_in,select_set= select_data_with_class(target_data,target_label,"Widar3.0",args.a,select_set) #选出来
    T_a_query=BaseCSIDataset(target_data_in,target_label_in,transfrom=args.transform)
    data_suppot, label_suppot, data_query, label_query=select_data_with_k_shot(target_data_not_in,target_label_not_in,args.k)
    T_n_support=BaseCSIDataset(data_suppot,label_suppot,transfrom=args.transform)
    T_n_query=BaseCSIDataset(data_query,label_query,transfrom=args.transform)
    ###################################处理target的数据########################################################
    return train_dataset_source_l,train_dataset_source_ul, val_dataset_source, test_dataset_source, T_a_query, T_n_support, T_n_query


from torch.utils.data import Dataset
import torch
class WidarDataset_new(Dataset):
    def __init__(self, file_path_list, label_list, transform=None, preload=False,a_select='0_4'):
        # 这里a_select是个字符串 
        self.file_path_list = file_path_list
        self.label_list = label_list
        self.transform = transform  
        self.preload = preload
        self.a_select=a_select

        if self.preload:
            print("⏫ 正在预加载数据到内存...")
            self.data = [np.load(p) for p in self.file_path_list]
            self.labels = [l for l in self.label_list]
            print("✅ 预加载完成")

        # 标签集合（升序去重）
        self.all_labels = torch.sort(torch.tensor(self.label_list).unique()).values.tolist()
    def get_all_labels(self):
        return self.all_labels  

    # 等待完善 其实就是替换个 文件后缀
        
    def __len__(self):
        return len(self.file_path_list)
    

    def __getitem__(self, idx):

        file_path = self.file_path_list[idx]
        label = self.label_list[idx]

        # 加载数据
        if self.preload:
            data_full = self.data[idx]  # shape: [6, 60, 1000]
        else:
            data_full = np.load(file_path)

        # 解析 a_select，例如 '0_1_2'
        try:
            channel_indices = [int(i) for i in self.a_select.split('_')]
        except:
            raise ValueError(f"a_select 参数格式错误: {self.a_select}")

        for ch in channel_indices:
            if ch < 0 or ch >= data_full.shape[0]:
                raise IndexError(f"a_select 中的通道索引 {ch} 超出范围，应在 0~{data_full.shape[0]-1} 之间")

        # 选择通道，返回 shape: [N, 60, 1000]
        data_select = data_full[channel_indices, :, :]  # shape: [N, 60, 1000]

        data_amp=data_select[:,0:30,:]
        data_phase=data_select[:,30:60,:]

        data = np.stack([data_amp, data_phase], axis=-1)  # shape: (N, 30, T, 2)
        # 转换为 Tensor
        # 现在是 天线 子载波 T 振幅/相位
        if self.transform:
            data = self.transform(data)
        else:
            data = torch.tensor(data, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.long)

        return data, label



def load_widar_data_uda_new(args, hparams):
    """
    加载Widar3.0数据集的UDA任务数据,只读取路径和标签
    返回训练集、验证集和测试集,使用自定义WidarDataset
    """
    root_dir = args.data_dir

    # 保持不变：获取源域和目标域的列表
    source_domain, target_domain = get_Widar_domain(args.target_domain, args.cross_domain_type, root_dir)

    # 修改点：只获取路径和标签
    source_paths, source_labels, target_paths, target_labels = get_widar_filelists_quick(
        source_domain, target_domain, root_dir, args.cross_domain_type
    )

    # 使用你待定义的 WidarDataset，输入为路径和标签
    source_dataset = WidarDataset_new(source_paths, source_labels,a_select=args.a_select)
    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])

    test_dataset = WidarDataset_new(target_paths, target_labels,a_select=args.a_select)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    # exit(0)
    # exit(0)
    
    return train_dataset, val_dataset, test_dataset
def load_widar_data_UDA_visual(args, hp):
    root_dir = args.data_dir

    # 获取 source 和 target 的域列表
    source_domain_list, target_domain_list = get_Widar_domain(
        args.target_domain,
        args.cross_domain_type,
        root_dir
    )
    print("Source domains:", source_domain_list)
    print("Target domains:", target_domain_list)
    # exit(0)

    # 获取 source 和 target 的路径及标签
    source_img, source_label, target_img, target_label = get_widar_filelists_quick(
        source_domain_list,
        target_domain_list,
        root_dir,
        args.cross_domain_type
    )

    source_dataset = visual_dataset(source_img, source_label)
    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])

    test_dataset = visual_dataset(target_img, target_label)

    print("📦 训练集样本数:", len(train_dataset))
    print("🧪 验证集样本数:", len(val_dataset))
    print("🧭 测试集样本数:", len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def load_widar_data_UDA_visual_1d(args, hp):
    root_dir = args.data_dir

    # 获取 source 和 target 的域列表
    source_domain_list, target_domain_list = get_Widar_domain(
        args.target_domain,
        args.cross_domain_type,
        root_dir
    )
    print("Source domains:", source_domain_list)
    print("Target domains:", target_domain_list)
    # exit(0)

    # 获取 source 和 target 的路径及标签
    source_img, source_label, target_img, target_label = get_widar_filelists_quick(
        source_domain_list,
        target_domain_list,
        root_dir,
        args.cross_domain_type
    )

    source_dataset = visual_dataset_1d(source_img, source_label)
    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])

    test_dataset = visual_dataset_1d(target_img, target_label)

    print("📦 训练集样本数:", len(train_dataset))
    print("🧪 验证集样本数:", len(val_dataset))
    print("🧭 测试集样本数:", len(test_dataset))
    
    return train_dataset, val_dataset, test_dataset


def load_widar_data_uda_DG(args, hparams):
    """
    加载Widar3.0数据集的UDA任务数据,只读取路径和标签
    和之前不同，这个是为DG任务设计的
    返回训练集、验证集和测试集,使用自定DGDataset
    """
    root_dir = args.data_dir

    # 保持不变：获取源域和目标域的列表
    source_domain, target_domain = get_Widar_domain(args.target_domain, args.cross_domain_type, root_dir)

    # 修改点：只获取路径和标签
    source_paths, source_labels, target_paths, target_labels = get_widar_filelists_quick(
        source_domain, target_domain, root_dir, args.cross_domain_type
    )

    # 使用你待定义的WidarDataset，输入为路径和标签
    source_dataset = DGDataset(source_paths, source_labels,config=args)
    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])

    test_dataset = DGDataset(target_paths, target_labels,config=args)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def load_widar_data_uda_DG_final(args, hparams):
    """
    加载Widar3.0数据集的UDA任务数据,只读取路径和标签
    和之前不同，这个是为DG任务设计的
    返回训练集、验证集和测试集,使用自定DGDataset
    """
    root_dir = args.data_dir

    # 保持不变：获取源域和目标域的列表
    source_domain, target_domain = get_Widar_domain(args.target_domain, args.cross_domain_type, root_dir)
    # 修改点：只获取路径和标签
    source_paths, source_labels, target_paths, target_labels = get_widar_filelists_quick(
        source_domain, target_domain, root_dir, args.cross_domain_type
    )

    # 使用你待定义的WidarDataset，输入为路径和标签
    train_paths, val_paths, train_labels, val_labels = train_test_split(
    source_paths, source_labels, test_size=0.2, random_state=0, shuffle=True
    )

    train_dataset = LiSADataset(train_paths, train_labels,config=args)
    val_dataset = LiSADataset(val_paths, val_labels,config=args)
    test_dataset = LiSADataset(target_paths, target_labels,config=args)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    # exit(0)
    
    return train_dataset, val_dataset, test_dataset

def load_widar_data_DG_DARC(args, hparams):
    """
    为DG任务设计的数据集，除了返回每个样本的标签外还会返回每个样本的域标签
    返回训练集、验证集和测试集,使用自定DGDataset
    """
    root_dir = args.data_dir

    # 保持不变：获取源域和目标域的列表
    source_domain, target_domain = get_Widar_domain(args.target_domain, args.cross_domain_type, root_dir)
    # 修改点：只获取路径和标签
    source_paths, source_labels, target_paths, target_labels = get_widar_filelists_quick(
        source_domain, target_domain, root_dir, args.cross_domain_type
    )

    # 使用你待定义的WidarDataset，输入为路径和标签
    train_paths, val_paths, train_labels, val_labels = train_test_split(
    source_paths, source_labels, test_size=0.2, random_state=0, shuffle=True
    )

    train_dataset = DARCDataset(train_paths, train_labels, args.cross_domain_type,config=args)
    val_dataset = DARCDataset(val_paths, val_labels, args.cross_domain_type,config=args)
    test_dataset = DARCDataset(target_paths, target_labels, args.cross_domain_type,config=args)




    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    # exit(0)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    import argparse
    file_dir = "/data0/zzy25/Signfi/dataset_home_276.mat"
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default="/data0/zzy25/widar3/no_process_this/")#"/mnt/ssd1/LiuSJ/") #数据的地址 
    parser.add_argument('--data-path', type=str, default='./data') # 数据地址  
    parser.add_argument('--csidataset', type=str, default='SignFi')#'Widar3'#'CSIDA',#'SignFi'  数据集有哪些  
    parser.add_argument('--backbone', type=str, default="CSIResNet") #使用的 backbone
    parser.add_argument("--evalmode",default="fast",help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")

    #zzy加的 
    parser.add_argument("--task_setting", default="UDA", help="various experiment setting")
    parser.add_argument("--cross_domain_type", default="room", help="set cross domain type")
    

    parser.add_argument('--data_type', type=str, default="amp+pha")
    parser.add_argument('--source_domain', type=str, default=None)
    parser.add_argument('--target_domain', type=str, default='user1')
    parser.add_argument('--a', type=int, default=4, help='shared classes')
    parser.add_argument('--k', type=int, default=5, help='shots per class')
    parser.add_argument('--k_train',type=int, default=5, help='k shots per class in source domain')
    parser.add_argument('--n', type=int, default=5, help='novel classes in target')
    # parser.add_argument("--FDA_hp", default=[(10,10),(10,50),(10,20),(10,30),(10,40),(10,60),(10,70),(10,80),(10,100),(10,200),(10,300),(10,400),(10,1000)], type=ast.literal_eval, help="test bandwith in FDA")

    # parser.add_argument('--stage', type=str, required=True, choices=['pretrain', 'proto', 'eval'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--results_file', type=str, default="test_results.txt")
    parser.add_argument("--model_save", default=500, type=int, help="Model save start step")
    parser.add_argument('--checkpoint_freq', type=int, default=5 ) # 每几轮保存
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    # check_signfi_stat(file_dir)
    # load_widar_data_uda(args, None)
    load_widar_data_uda_DG_final(args, None)


    args = parser.parse_args()
    # check_signfi_stat(file_dir)
    # load_widar_data_uda(args, None)
    load_widar_data_uda_DG_final(args, None)


