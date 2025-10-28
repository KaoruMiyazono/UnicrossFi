# from utils import BaseCSIDataset ,get_valid_csida_files,select_data_with_class,select_data_with_k_shot,WiSDADataset #debug用 
from data.utils import BaseCSIDataset ,get_valid_csida_files,select_data_with_class,select_data_with_k_shot,WiSDADataset,WiGRUNT_dataset,Wiopen_dataset,LiSADataset_CSIDA

import os
import pickle
import numpy as np 
from torch.utils.data import ConcatDataset, random_split
from sklearn.model_selection import train_test_split
import re
from scipy.signal import firwin, filtfilt






def get_csida_domain(target_domain,cross_domain_type,root_dir):
    # 获得domain列表 
    all_room_incsida=['room0','room1']
    all_user_incsida=['user0','user1','user2','user3','user4',]
    all_location_incsida=['location0','location1','location2']
    source_domain_list = []
    target_domain_list = []

    #############################检查 防止 出现 错误的目标域输入#########################################################
    if cross_domain_type == 'room':
        if target_domain not in all_room_incsida:
            raise ValueError(f"target_domain '{target_domain}' 不在 room 域列表里！合法选项: {all_room_incsida}")
    elif cross_domain_type == 'user':
        if target_domain not in all_user_incsida:
            raise ValueError(f"target_domain '{target_domain}' 不在 user 域列表里！合法选项: {all_user_incsida}")
    elif cross_domain_type == 'location':
        if target_domain not in all_location_incsida:
            raise ValueError(f"target_domain '{target_domain}' 不在 location 域列表里！合法选项: {all_location_incsida}")
    else:
        raise ValueError(f"cross_domain_type '{cross_domain_type}' 不合法！应该是 'room'、'user' 或 'location' 中的一个")
    ############################检查 防止 出现 错误的目标域输入#########################################################

    ##############################生成源域和目标域列表#################################################################
    all_files=get_valid_csida_files(root_dir) #找到对应目录下的所有文件 
    target_domain_formatted = target_domain.replace('room', 'room_').replace('user', 'user_').replace('location', 'loc_')

    # 只在完全匹配这个名字时跳过
    skip_exact = 'room_0_user_1_loc_1'
    for prefix in all_files:
        # if prefix == skip_exact:
        #     print(f"跳过特定文件: {prefix}")
        #     continue  
        if target_domain_formatted in prefix:
            target_domain_list.append(prefix)
        else:
            source_domain_list.append(prefix)
    print(f'source_domain_list如下：{source_domain_list}')
    print(f'target_domain_list如下：{target_domain_list}')
    ##############################生成源域和目标域列表#################################################################
    return source_domain_list,target_domain_list

def load_csidadata_WiGRUNT(source_domain_list,target_domain_list,root_dir):
# /opt/data/private/ablation_study/data_wigrut/ges_0_room_1_loc_0_user_1_cnt_0.png
    print("🚀 正在加载数据...")
    source_img_list ,source_label_list = [],[]
    target_img_list,target_label_list = [], []
    files = os.listdir(root_dir)
    for domain_prefix in source_domain_list: #room_0_loc_0_user_0
        domain_split=domain_prefix.split('_')
        string_domain=f'room_{domain_split[1]}_loc_{domain_split[3]}_user_{domain_split[5]}'
        print(string_domain)

        
        for file in files:
            file_split=file.split('_')
            ges=file_split[1]
            room=file_split[3]
            loc=file_split[5]
            user=file_split[7]
            string_this_file=f'room_{room}_loc_{loc}_user_{user}'
            # print(string_this_file)
            # exit(0)
            if string_this_file == string_domain:
                source_img_list.append(os.path.join(root_dir, file))
                source_label_list.append(int(ges))
                
    for domain_prefix in target_domain_list: #room_0_loc_0_user_0
        domain_split=domain_prefix.split('_')
        string_domain=f'room_{domain_split[1]}_loc_{domain_split[3]}_user_{domain_split[5]}'
        
        for file in files:
            file_split=file.split('_')
            ges=file_split[1]
            room=file_split[3]
            loc=file_split[5]
            user=file_split[7]
            string_this_file=f'room_{room}_loc_{loc}_user_{user}'
            if string_this_file == string_domain:
                target_img_list.append(os.path.join(root_dir, file))
                target_label_list.append(int(ges))
    return source_img_list,source_label_list, target_img_list,target_label_list
            
        
    


def load_csidadata_Wisda(source_domain_list,target_domain_list,root_dir):
    
    # 初始化 
    print("🚀 正在加载数据...")
    
    source_img_list ,source_label_list = [],[]
    target_img_list,target_label_list = [], []
    
    # 处理源域文件 
    files = os.listdir(root_dir)
    for domain_prefix in source_domain_list: #room_0_loc_0_user_0
        # gesture_5_room_1_loc_1_user_1_at_2_sub_36_cnt_111.png  ['room', '0', 'loc', '0', 'user', '0']
        # print(domain_prifix)
        domain_split=domain_prefix.split('_')


        string_domain=f'room_{domain_split[1]}_loc_{domain_split[3]}_user_{domain_split[5]}'
        # print(string_domain)
        for file in files:
            # print(file)
            file_split= file.split('_')
            gesture_name = file_split[1]
            room_name =file_split[3]
            loc_name = file_split[5]
            user_name = file_split[7]
            # print(f"room_name: {room_name}, loc_name: {loc_name}, user_name: {user_name}")
            string_this_file=f'room_{room_name}_loc_{loc_name}_user_{user_name}'
            if string_this_file == string_domain:
                source_img_list.append(os.path.join(root_dir, file))
                source_label_list.append(int(gesture_name))

    # print(len(source_img_list))    
    
    
    # 处理目标域数据 
    for domain_prefix in target_domain_list: #room_0_loc_0_user_0
        # gesture_5_room_1_loc_1_user_1_at_2_sub_36_cnt_111.png  ['room', '0', 'loc', '0', 'user', '0']
        # print(domain_prifix)
        domain_split=domain_prefix.split('_')


        string_domain=f'room_{domain_split[1]}_loc_{domain_split[3]}_user_{domain_split[5]}'
        print(string_domain)
        for file in files:
            # print(file)
            file_split= file.split('_')
            gesture_name = file_split[1]
            room_name =file_split[3]
            loc_name = file_split[5]
            user_name = file_split[7]
            # print(f"room_name: {room_name}, loc_name: {loc_name}, user_name: {user_name}")
            string_this_file=f'room_{room_name}_loc_{loc_name}_user_{user_name}'
            if string_this_file == string_domain:
                target_img_list.append(os.path.join(root_dir, file))
                target_label_list.append(int(gesture_name))
    # print(len(target_img_list))
    return source_img_list,source_label_list, target_img_list,target_label_list
            
            
        # print(string_domain)
        
    
    


def load_csida_data_from_files(source_domain_list, target_domain_list, root_dir): 
    # 初始化列表
    print("🚀 正在加载数据...")
    source_amp_list, source_pha_list, source_label_list,source_domain_tags = [], [], [], []
    target_amp_list, target_pha_list, target_label_list,target_domain_tags = [], [], [], []

    # 处理源域文件
    for domain_prefix in source_domain_list:
        # 构建文件路径
        amp_path = os.path.join(root_dir, domain_prefix + "_CSIDA_amp.pkl")
        pha_path = os.path.join(root_dir, domain_prefix + "_CSIDA_pha.pkl")
        label_path = os.path.join(root_dir, domain_prefix + "_CSIDA_label.pkl")

        # 读取文件
        if os.path.exists(amp_path):
            with open(amp_path, 'rb') as f:
                data = pickle.load(f)
                source_amp_list.append(data)
                source_domain_tags.extend([domain_prefix] * data.shape[0])  # 为每条数据添加对应的域
        if os.path.exists(pha_path):
            with open(pha_path, 'rb') as f:
                data = pickle.load(f)
                source_pha_list.append(data)
        if os.path.exists(label_path):
            with open(label_path, 'rb') as f:
                data = pickle.load(f)
                source_label_list.append(data)

    # 处理目标域文件
    for domain_prefix in target_domain_list:
        # 构建文件路径
        amp_path = os.path.join(root_dir, domain_prefix + "_CSIDA_amp.pkl")
        pha_path = os.path.join(root_dir, domain_prefix + "_CSIDA_pha.pkl")
        label_path = os.path.join(root_dir, domain_prefix + "_CSIDA_label.pkl")

        # 读取文件
        if os.path.exists(amp_path):
            with open(amp_path, 'rb') as f:
                data = pickle.load(f)
                target_amp_list.append(data)
                target_domain_tags.extend([domain_prefix] * data.shape[0])  # 为每条数据添加对应的域
        if os.path.exists(pha_path):
            with open(pha_path, 'rb') as f:
                data = pickle.load(f)
                target_pha_list.append(data)
        if os.path.exists(label_path):
            with open(label_path, 'rb') as f:
                data = pickle.load(f)
                target_label_list.append(data)

    # 将列表转换为 numpy 数组
    # for i, amp_array in enumerate(source_amp_list):
    #     print(f"元素 {i} 的形状: {amp_array.shape}")
    # print(source_label_list[1])
    source_amp_array =  np.concatenate(source_amp_list,axis=0)
    source_pha_array =  np.concatenate(source_pha_list,axis=0)
    source_label_array =  np.concatenate(source_label_list,axis=0)
    # print(source_label_array[113:226])
    target_amp_array =  np.concatenate(target_amp_list,axis=0)
    target_pha_array =  np.concatenate(target_pha_list,axis=0)
    target_label_array =  np.concatenate(target_label_list,axis=0)

    # TODO 对amp 和phase组合数据进行低频滤波
    # 1. 滤波器设计 (与之前相同)
    fs = 1000  # 采样频率
    nyquist_rate = fs / 2.0
    cutoff_freq = 1.0
    numtaps = 101
    normalized_cutoff = cutoff_freq / nyquist_rate
    fir_coeff = firwin(numtaps, normalized_cutoff, pass_zero=False, window='hamming')
    source_amp_array = filtfilt(fir_coeff, [1.0], source_amp_array, axis=-1)
    target_amp_array = filtfilt(fir_coeff, [1.0], target_amp_array, axis=-1)

    # 对amp array进行归一化
    amp_min = np.min(source_amp_array)
    amp_max = np.max(source_amp_array)
    source_amp_array = (source_amp_array - amp_min) / (amp_max - amp_min + 1e-8)
    source_amp_array = np.pi*source_amp_array
    amp_min = np.min(target_amp_array)
    amp_max = np.max(target_amp_array)
    target_amp_array = (target_amp_array - amp_min) / (amp_max - amp_min + 1e-8)
    target_amp_array = np.pi*target_amp_array

    print(f"源域数据大小: {source_amp_array.shape}, {source_pha_array.shape}, {source_label_array.shape}")
    print(f"目标域数据大小: {target_amp_array.shape}, {target_pha_array.shape}, {target_label_array.shape}")

    print("🚀 数据加载完成！")

    # 返回结果
    return source_amp_array, source_pha_array, source_label_array, target_amp_array, target_pha_array, target_label_array,source_domain_tags,target_domain_tags




def load_csida_data_uda(args,hparams):

    root_dir=args.data_dir #数据地址 
    

    source_domain_list,target_domain_list=get_csida_domain(args.target_domain,args.cross_domain_type,root_dir) #得到 所有domain的列表  
    source_amp_array, source_pha_array, source_label_array, target_amp_array, target_pha_array, target_label_array,source_domain_list,target_domain_list=load_csida_data_from_files(source_domain_list,target_domain_list,root_dir) #加载数据
    assert (source_amp_array.shape[0]==len(source_domain_list))
    assert (target_amp_array.shape[0]==len(target_domain_list))



    # 然后在最后一个维度上拼接
    source_concatenated_array = np.concatenate((np.expand_dims(source_amp_array, axis=-1), np.expand_dims(source_pha_array, axis=-1)), axis=-1)
    target_concatenated_array = np.concatenate((np.expand_dims(target_amp_array, axis=-1), np.expand_dims(target_pha_array, axis=-1)), axis=-1)
    # print(source_concatenated_array.shape)

    train_sample, val_sample, train_label, val_label, train_domain, val_domain = train_test_split(
        source_concatenated_array,
        source_label_array,
        source_domain_list,
        test_size=0.2,
        random_state=0,
        shuffle=True
    )

    # 包装成 Dataset
    train_dataset = BaseCSIDataset(train_sample, train_label, domain=train_domain, descriptation=args.task_setting+'_cross'+args.cross_domain_type+'_'+args.data_type+'source_domain')
    val_dataset = BaseCSIDataset(val_sample, val_label, domain=val_domain, descriptation=args.task_setting+'_cross'+args.cross_domain_type+'_'+args.data_type+'source_domain')
    
    test_dataset=BaseCSIDataset(target_concatenated_array, target_label_array,domain=target_domain_list,descriptation=args.task_setting+'_cross'+args.cross_domain_type+'_'+args.data_type+'target_domain')
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
    print("Number of test samples:", len(test_dataset))
    return train_dataset, val_dataset, test_dataset

def load_csida_data_UDA_WiSDA(args,hparams):
    "为WiSDA写的CSIDA加载函数"
    root_dir=args.data_dir #地址 
    source_domain_list,target_domain_list = get_csida_domain(args.target_domain,args.cross_domain_type,"/opt/data/common/default/wifidata/csida/CSI_301/") 
    source_img,source_label,target_img,target_label= load_csidadata_Wisda(source_domain_list,target_domain_list,root_dir)
    
    source_dataset=WiSDADataset(source_img,source_label)
    train_size = int(0.8 * len(source_dataset))  # 80% 作为训练集
    val_size = len(source_dataset) - train_size  # 剩下的 20% 作为验证集
    train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])
    # img, label = source_dataset[0]
    test_dataset= WiSDADataset(target_img,target_label)
    
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
    print("Number of test samples:", len(test_dataset))
    
    return train_dataset, val_dataset, test_dataset

def load_csida_data_UDA_WiGRUNT(args,hp):
    root_dir=args.data_dir
    source_domain_list,target_domain_list = get_csida_domain(args.target_domain,args.cross_domain_type,"/opt/data/common/default/wifidata/csida/CSI_301/")
    print(source_domain_list)
    source_img,source_label,target_img,target_label=load_csidadata_WiGRUNT(source_domain_list,target_domain_list,root_dir)  

    source_dataset=WiGRUNT_dataset(source_img,source_label)
    
    train_size = int(0.8 * len(source_dataset))  # 80% 作为训练集
    val_size = len(source_dataset) - train_size  # 剩下的 20% 作为验证集
    train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])
    # img, label = source_dataset[0]
    test_dataset= WiGRUNT_dataset(target_img,target_label)
    
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
    print("Number of test samples:", len(test_dataset))
    
    return train_dataset, val_dataset, test_dataset


def load_csida_data_UDA_Wiopen(args,hp):
    root_dir=args.data_dir
    source_domain_list,target_domain_list = get_csida_domain(args.target_domain,args.cross_domain_type,"/opt/data/common/default/wifidata/csida/CSI_301/")
    print(source_domain_list)
    source_img,source_label,target_img,target_label=load_csidadata_WiGRUNT(source_domain_list,target_domain_list,root_dir)  
    ########################这里改变了@####################################
    # source_dataset=Wiopen_dataset(source_img,source_label)

    # # img,dfs,label=source_dataset[0]

    
    # train_size = int(0.8 * len(source_dataset))  # 80% 作为训练集
    # val_size = len(source_dataset) - train_size  # 剩下的 20% 作为验证集
    # train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])
    # # img, label = source_dataset[0]
    # test_dataset= Wiopen_dataset(target_img,target_label)
    ########################这里改变了@####################################
    from sklearn.model_selection import train_test_split
    train_img, val_img, train_label, val_label = train_test_split(
    source_img, source_label, test_size=0.2,random_state=0
    )

    # 打印结果


    train_dataset=Wiopen_dataset(train_img,train_label)
    val_dataset=Wiopen_dataset(val_img,val_label)
    test_dataset= Wiopen_dataset(target_img,target_label)
    
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
    print("Number of test samples:", len(test_dataset))
    # exit(0)
    
    return train_dataset, val_dataset, test_dataset
    


def load_csida_data_OSFSUDA(args,hparams):
    root_dir=args.data_dir
    
    source_domain_list,target_domain_list=get_csida_domain(args.target_domain,args.cross_domain_type,root_dir) #得到 所有domain的列表
    source_amp_array, source_pha_array, source_label_array, target_amp_array, target_pha_array, target_label_array=load_csida_data_from_files(source_domain_list,target_domain_list,root_dir) #加载数据
    # 然后在最后一个维度上拼接
    source_concatenated_array = np.concatenate((np.expand_dims(source_amp_array, axis=-1), np.expand_dims(source_pha_array, axis=-1)), axis=-1)
    target_concatenated_array = np.concatenate((np.expand_dims(target_amp_array, axis=-1), np.expand_dims(target_pha_array, axis=-1)), axis=-1)
    # print(source_concatenated_array.shape)
    
    
    ##################################处理source数据############################################################################
    source_data , source_label,_,_,_= select_data_with_class(source_concatenated_array,source_label_array,dataset_name='CSIDA') #选出来
    train_source_data,test_source_data,train_source_labels,test_source_labels = train_test_split(source_data, source_label, test_size=0.2, random_state=args.seed)
    print(train_source_data.shape)
    print(train_source_labels.shape)
    test_dataset_source=BaseCSIDataset(test_source_data, test_source_labels,transfrom=args.transform) # source的测试集 
    train_source_data,val_source_data,train_source_labels,val_source_labels = train_test_split(train_source_data, train_source_labels, test_size=0.2, random_state=args.seed) # 80% 作为训练集
    
    print(train_source_data.shape)
    print(train_source_labels.shape)
    val_dataset_source=BaseCSIDataset(val_source_data, val_source_labels,transfrom=args.transform) # source的验证集
    train_support, train_label_support, train_query, train_label_query=select_data_with_k_shot(train_source_data,train_source_labels,args.k_train *2 ) # 选择k-shot的支持集和查询集
    #print(train_support.shape)
    #print(train_label_support)
    train_dataset_source_l=BaseCSIDataset(train_support, train_label_support,transfrom=args.transform)
    train_dataset_source_ul=BaseCSIDataset(train_query, train_label_query,transfrom=args.transform)
    
    # print(source_data.shape)
    # exit(0)
    ####################################处理source数据##########################################################################
    
    
    ####################################处理target数据##########################################################################
    target_data_in,target_label_in,target_data_not_in,target_label_not_in,select_set=select_data_with_class(target_concatenated_array,target_label_array,dataset_name='CSIDA') #选出来 
    T_a_query=BaseCSIDataset(target_data_in,target_label_in,domain=target_domain_list,descriptation=args.task_setting+'_cross'+args.cross_domain_type+'_'+args.data_type+'target_domain_a_query',transfrom=args.transform)
    data_suppot, label_suppot, data_query, label_query=select_data_with_k_shot(target_data_not_in,target_label_not_in,args.k)
    T_n_support=BaseCSIDataset(data_suppot,label_suppot,domain=target_domain_list,descriptation=args.task_setting+'_cross'+args.cross_domain_type+'_'+args.data_type+'target_domain_n_support',transfrom=args.transform)
    T_n_query=BaseCSIDataset(data_query,label_query,domain=target_domain_list,descriptation=args.task_setting+'_cross'+args.cross_domain_type+'_'+args.data_type+'target_domain_n_query',transfrom=args.transform)
    

    # 打印每个数据集的大小
    print(f"Train dataset (labeled) size: {len(train_dataset_source_l)}")
    print(f"Train dataset (unlabeled) size: {len(train_dataset_source_ul)}")
    print(f"Validation dataset size: {len(val_dataset_source)}")
    print(f"Test dataset size: {len(test_dataset_source)}")
    ####################################处理target数据##########################################################################
    return  train_dataset_source_l,train_dataset_source_ul,val_dataset_source,test_dataset_source,T_a_query,T_n_support,T_n_query



def load_csida_data_DG(args,hparams):

    root_dir=args.data_dir #数据地址 
    

    source_domain_list,target_domain_list=get_csida_domain(args.target_domain,args.cross_domain_type,root_dir) #得到 所有domain的列表  
    source_amp_array, source_pha_array, source_label_array, target_amp_array, target_pha_array, target_label_array,source_domaintag_list,target_domaintag_list=load_csida_data_from_files(source_domain_list,target_domain_list,root_dir) #加载数据

    #  现在我已经能拿到明确的domain信息了 domain信息是一个列表 我要重新包装一下dataset


    # 然后在最后一个维度上拼接
    source_concatenated_array = np.concatenate((np.expand_dims(source_amp_array, axis=-1), np.expand_dims(source_pha_array, axis=-1)), axis=-1)
    target_concatenated_array = np.concatenate((np.expand_dims(target_amp_array, axis=-1), np.expand_dims(target_pha_array, axis=-1)), axis=-1)
    # print(source_concatenated_array.shape)

    # source_dataset=LiSADataset_CSIDA(source_concatenated_array, source_label_array,domain=source_domaintag_list)
    # train_size = int(0.8 * len(source_dataset))  # 80% 作为训练集
    # val_size = len(source_dataset) - train_size  # 剩下的 20% 作为验证集
    # train_dataset, val_dataset = random_split(source_dataset, [train_size, val_size])
    train_sample, val_sample, train_label, val_label, train_domain, val_domain = train_test_split(
    source_concatenated_array,
    source_label_array,
    source_domaintag_list,
    test_size=0.2,
    random_state=0,
    shuffle=True  # 默认是 True，这里显式写出来更清晰
    )
    train_dataset = LiSADataset_CSIDA(train_sample, train_label, domain=train_domain)
    val_dataset = LiSADataset_CSIDA(val_sample, val_label, domain=val_domain)
    test_dataset=LiSADataset_CSIDA(target_concatenated_array, target_label_array,domain=target_domaintag_list)
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
    print("Number of test samples:", len(test_dataset))

    # exit(0)
    return train_dataset, val_dataset, test_dataset

    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default="/data1/zzy25/CSIDA_WISDA")#"/mnt/ssd1/LiuSJ/") #数据的地址  /data0/zzy25/CSIDA/CSIDA/CSI_301/
    parser.add_argument('--data-path', type=str, default='./data') # 数据地址  
    parser.add_argument('--csidataset', type=str, default='CSIDA')#'Widar3'#'CSIDA',#'SignFi'  数据集有哪些  
    parser.add_argument('--backbone', type=str, default="CSIResNet") #使用的 backbone
    parser.add_argument("--evalmode",default="fast",help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")

    #zzy加的 
    parser.add_argument("--task_setting", default="UDA", help="various experiment setting")
    parser.add_argument("--cross_domain_type", default="room", help="set cross domain type")
    

    parser.add_argument('--data_type', type=str, default="amp+pha")
    parser.add_argument('--source_domain', type=str, default=None)
    parser.add_argument('--target_domain', type=str, default='room1')
    parser.add_argument('--a', type=int, default=5, help='shared classes')
    parser.add_argument('--k', type=int, default=5, help='shots per class')
    parser.add_argument('--k_train', type=int, default=2, help='shots per class')
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

    hparams = [2,3]  # 这里可以置空

    # train_datatset,val_dataset,test_datatset= load_csida_data_uda(args, hparams)
    # _,_,_,_,_,_=load_csida_data_OSFSUDA(args,hparams)
    x=load_csida_data_UDA_WiSDA(args,hparams)




