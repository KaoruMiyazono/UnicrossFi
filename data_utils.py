import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
# from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d
import torch
import re

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import numpy as np

from tqdm import tqdm

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'/home/caizhicheng/zzy/ts2vec/datasets/UEA/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'/home/caizhicheng/zzy/ts2vec/datasets/UEA/Multivariate_arff/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1])) #计算每一列的均值 
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y) #返回的是 所有的label对应的类别列表 
    return train_X, train_y, test_X, test_y

def conj_multi(csi):
    # csi shape = [N,T,C,A] [N,T,30,3]
    ref_ant = np.expand_dims(np.conjugate(csi[:,:,:,0]),axis=3)
    conjed_phase = np.angle(csi*ref_ant)[:,:,:,[1,2]]
    return conjed_phase

def csi_process(data): 
    """
    #对输入进行 共轭乘法
        input: data:(N,T,30,3,4)
       output :(N,T,30,2,2) #
    """
    data_amp,data_ang=data[:,:,:,:,0],data[:,:,:,:,1] #得到相位和振幅信息 

    #还原复数矩阵
    real_part = data_amp * np.cos(data_ang)
    imag_part = data_amp * np.sin(data_ang)
    csi = real_part + 1j * imag_part

    conj_phase=conj_multi(csi)
    conj_phase=np.unwrap(conj_phase)
    amp_return=data_amp[:,:,:,[1,2]] #因为共轭乘法更多对 相位起作用，所以这里直接用了运算前对应的振幅

    result = np.stack((conj_phase, amp_return), axis=-1)  # shape will be [N, T, 30, 2, 2] #频道一起
    return result

def load_Widar():
    train_data = torch.load(f'/workspace/data0/wifi_datasets/datasets/widar/widar_all_r2/widar_r2_train.pt')
    test_data = torch.load(f'/workspace/data0/wifi_datasets/datasets/widar/widar_all_r2/widar_r2_test.pt')
    val_data = torch.load(f'/workspace/data0/wifi_datasets/datasets/widar/widar_all_r2/widar_r2_val.pt')
    
    def extract_data(data):
        res_data_torch,res_labels_torch=data.tensors #得到数据和标签
        res_data_np=np.array(res_data_torch) #(N,500,30,3,4)
        N,T,_,_,_=res_data_np.shape
        res_data_np_process=csi_process(res_data_np).reshape(N,T,-1) #这个reshape是要压变量
        return res_data_np_process, np.array(res_labels_torch)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    val_X, val_y = extract_data(val_data)
    
    #归一化步骤 
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1])) #计算每一列的均值 
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    val_X = scaler.transform(val_X.reshape(-1, val_X.shape[-1])).reshape(val_X.shape)
    
    #处理label的部分
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y) #所有的label对应的类别列表 
    val_y = np.vectorize(transform.get)(val_y) #所有的label对应的类别列表 
    return train_X, train_y, test_X, test_y,val_X,val_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data

# load_Widar()



def align_time_length(signal,time_stamp,newtimelen):
    """
    One principle followed during the process of time-axis alignment 
    is to preserve as much information from the original data as possible.

    input shape [T,C,A]
    """
    if signal.shape[0]<newtimelen: # interpolation
        return updample(signal,time_stamp,newtimelen)
    if signal.shape[0]==newtimelen:
        return signal
    if signal.shape[0]>newtimelen: # using direct deletion followed by approximation of points
        return downsample(signal,time_stamp,newtimelen)

def updample(data,timestamp,exp_length):
    # input data shape [T,C,A]
    antenna_num = data.shape[-1]

    # generate the new time stamp
    interp_functions = [interp1d(timestamp, data[:, :, i], kind='linear', axis=0, fill_value='extrapolate') for i in range(antenna_num)]

    #interp_function = interp1d(timestamp,data, kind='linear', fill_value='extrapolate')

    new_timestamps = np.linspace(timestamp.min(), timestamp.max(), num=exp_length-len(timestamp))
    new_timestamps = np.concatenate((timestamp, new_timestamps))
    new_timestamps = np.sort(new_timestamps)
    interpolated_data = np.stack([interp(new_timestamps) for interp in interp_functions], axis=-1) # shape (T, C, A)
    return interpolated_data


def downsample(data,time_stamp,exp_length):
    #data shape should be [T,C,A]
    #Calculate new time intervals based on time stamps and the expected new time length.
    #Only one sample per time interval; achieved by taking individual point or averaging.

    #Calculate the average time interval for the new length, 
    #which is equivalent to averaging or using another method to combine the sample points within this time interval into a single point.
    intervel = (time_stamp[-1] - time_stamp[0]) / exp_length 
    cur_range = time_stamp[0] + intervel
    temp_list = []
    align_data = [data[0,:,:]] # init list with data at i=0
    nearest_i = -1

    for i in range(1,len(time_stamp)):  #Iterate through all sampling points.
        # If the current length of align_data reaches the last length, do not proceed to the next one directly; 
        #instead, combine all the remaining points into a single point.
        if len(align_data) == exp_length-1:
            align_data.append(data[-1,:,:])
            break
        #Count the elements between time[i] and time[i]+interval.
        if time_stamp[i]> len(align_data)*intervel: # If the time_stamp is outside the interval, then insert the nearest element to the interval.
            if len(temp_list) !=0:
                align_data.append(data[temp_list[-1],:,:])
            else:#If there are no values in the tmp list, insert the current value instead.
                align_data.append(data[i,:,:])
            temp_list=[]
        else:#If the time_stamp is within the interval, add the current point as the closest point.
            temp_list.append(i)

    if len(align_data) < exp_length:
        
        #Exclude the last element of align_data (to avoid duplication) and add the remaining elements.
        align_data = align_data[:-1]
        additional_number =  exp_length-len(align_data)
        tmp = data[len(time_stamp)-additional_number:,:,:]
        for i in range(additional_number):
            align_data.append(tmp[i,:,:])
        print("shorter than new_length, add the last element")
    align_data = np.stack(align_data)
    return align_data

def conj_multi_falldefi(csi):
        # csi shape = [T,C,A] [T,30,3]
        ref_ant = np.expand_dims(np.conjugate(csi[:,:,0]),axis=2)
        conjed_phase = np.angle(csi*ref_ant)[:,:,[1,2]]
        return conjed_phase
def csi_process_falldefi(csi,stamp):
    amp_csi=np.abs(csi) #得到振幅
    amp_csi=torch.from_numpy(align_time_length(amp_csi,stamp,500)) #对齐振幅
    amp_csi=amp_csi[:,:,1:,:]

    conj_phase=conj_multi_falldefi(csi)
    phase_csi=np.unwrap(conj_phase)
    phase_csi = torch.from_numpy(align_time_length(phase_csi,stamp,500)) # #对齐相位
    result = np.concatenate((phase_csi, amp_csi), axis=-1)  # 在最后一个维度拼接
    # print(result.shape)
    T,C,A,_=result.shape

    result=result.transpose(2,1,0,3)
    # print(result.shape)
    # exit(0)
    return result
    # return result.reshape(T,-1)


def load_Falldefi():
    #读数据
    file_path_falldef='/opt/data/common/default/wifidata/falldefi/falldefi_raw_complex.pkl'
    f = open(file_path_falldef,'rb')
    pkl_data = pickle.load(f)
    csi  = pkl_data['record']       # a list of complex signal, length is equal to the number of samples
    label = pkl_data['label']       # a list, length is equal to the number of samples
    time_stamps = pkl_data['stamp'] # a list

    #设置随机种子并 打乱样本
    total_index = list(range(len(label)))
    random.seed(42)
    random.shuffle(total_index)

    split_ratio = 0.8
    split_index = int(len(total_index) * split_ratio)
    trainval_index_list = total_index[:split_index] 
    test_index = total_index[split_index:]
    # train/val split 9:1
    split_ratio = 0.9
    split_index = int(len(trainval_index_list) * split_ratio)
    train_index = trainval_index_list[:split_index]
    val_index = trainval_index_list[split_index:]


    #train的一部分
    csi_train  = [csi[i] for i in train_index]      # a list of complex signal, length is equal to the number of samples
    label_train = [label[i] for i in train_index]       # a list, length is equal to the number of samples
    time_stamps_train = [time_stamps[i] for i in train_index] # a list


    #test的一部分
    csi_test  = [csi[i] for i in test_index]      # a list of complex signal, length is equal to the number of samples
    label_test = [label[i] for i in test_index]       # a list, length is equal to the number of samples
    time_stamps_test = [time_stamps[i] for i in test_index] # a list

    #val的一部分
    csi_val  = [csi[i] for i in val_index]      # a list of complex signal, length is equal to the number of samples
    label_val = [label[i] for i in val_index]       # a list, length is equal to the number of samples
    time_stamps_val = [time_stamps[i] for i in val_index] # a list

    def extract_data(csi,label,time_stamps):
        # print(csi.shape)
        csi_list=[]
        for i,csi_single in enumerate(csi):
            csi_sp=csi_process_falldefi(csi[i],time_stamps[i]) #处理得到的csi数据
            csi_list.append(csi_sp)
        
        csi_np=np.array(csi_list, dtype=np.float32)
        lable_np=np.array(label,dtype=np.int64)

        print(csi_np.shape)
        print(lable_np.shape)

        return csi_np,lable_np

        # print(label.shape)
        # print(time_stamps.shape)
    train_X, train_y=extract_data(csi_train,label_train,time_stamps_train)
    test_X,test_y=extract_data(csi_test,label_test,time_stamps_test)
    val_X,val_y=extract_data(csi_val,label_val,time_stamps_val)
    # exit(0)
    #归一化步骤 
    # scaler = StandardScaler()
    # scaler.fit(train_X.reshape(-1, train_X.shape[-1])) #计算每一列的均值 
    # train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    # test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    # val_X = scaler.transform(val_X.reshape(-1, val_X.shape[-1])).reshape(val_X.shape)
    
    #处理label的部分
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y) #所有的label对应的类别列表 
    val_y = np.vectorize(transform.get)(val_y) #所有的label对应的类别列表 
    return train_X, train_y, test_X, test_y,val_X,val_y
# load_Falldefi()

def extract_data_from_caida():
    """1"""
    data_prefix = "/root/autodl-tmp/CSI_301/"
    all_data = {
    'roomid': [],
    'locid': [],
    'userid': []
}
    file_pattern = re.compile(r'room_(\d+)_loc_(\d+)_user_(\d+)_CSIDA_.*\.pkl')
    processed_combinations = set()
    for filename in os.listdir(data_prefix):
    # 匹配符合规则的文件
        match = file_pattern.match(filename)
        if match:
            roomid = int(match.group(1))
            locid = int(match.group(2))
            userid = int(match.group(3))
            # file_type = match.group(4)
            
            # 使用 (roomid, locid, userid) 来判断是否已经处理过这个组合
            if (roomid, locid, userid) not in processed_combinations:
                # 将该组合加入已处理的集合
                processed_combinations.add((roomid, locid, userid))
                
                # 将提取的 roomid, locid, userid 添加到 all_data 中
                all_data['roomid'].append(roomid)
                all_data['locid'].append(locid)
                all_data['userid'].append(userid)
                # print(f"Processed combination: room_{roomid}_loc_{locid}_user_{userid}")
    all_sel_amp = []
    all_sel_pha = []
    all_sel_gesture = []
    for roomid, locid, userid in zip(all_data['roomid'], all_data['locid'], all_data['userid']):
        f=open(data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_amp.pkl','rb')
        # print('extracting:',data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_amp.pkl')
        sel_amp=pickle.load(f)    # np.array (N,A,C,T) (113, 3, 114, 1800)
        f.close()
        f=open(data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_pha.pkl','rb')
        # print('extracting:',data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_pha.pkl')
        sel_pha=pickle.load(f)  # np.array (N,A,C,T) (113, 3, 114, 1800)
        f.close()
        f=open(data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_label.pkl','rb')
        sel_gesture = pickle.load(f) # np.array (N,) (113,)
        f.close()  
        all_sel_amp.append(sel_amp)
        all_sel_pha.append(sel_pha)
        all_sel_gesture.append(sel_gesture)


    # 拼接扩展后的数组
    all_sel_amp_combined = np.concatenate(all_sel_amp)  # 沿最后一个维度拼接
    all_sel_pha_combined = np.concatenate(all_sel_pha)
    all_sel_gesture_numpy=np.concatenate(all_sel_gesture)
    # print(all_sel_amp_combined.shape)
    # print(all_sel_pha_combined.shape)
    all_sel_pha_combined =all_sel_pha_combined[..., np.newaxis] 
    all_sel_amp_combined =all_sel_amp_combined[..., np.newaxis] 
    # all_sel_gesture_numpy = np.array(all_sel_gesture)

    combined = np.concatenate((all_sel_pha_combined, all_sel_amp_combined), axis=-1)
    N = combined.shape[0]
    split_seed = 42
    random.seed(split_seed)
    total_index = list(range(N))
    random.shuffle(total_index)

    split_ratio = 0.8
    split_index = int(N * split_ratio)
    trainval_index_list = total_index[:split_index]
    test_index = total_index[split_index:]

    # train/val split 9:1
    split_ratio = 0.9
    split_index = int(len(trainval_index_list) * split_ratio)
    train_index = trainval_index_list[:split_index]
    val_index = trainval_index_list[split_index:]

    # 使用切片来获取train、val、test数据
    train_data = combined[train_index]  # shape (train_size, 114, 3, 1800, 2)
    val_data = combined[val_index]      # shape (val_size, 114, 3, 1800, 2)
    test_data = combined[test_index]    # shape (test_size, 114, 3, 1800, 2)


    train_label=all_sel_gesture_numpy[train_index]
    test_label=all_sel_gesture_numpy[test_index]
    val_label=all_sel_gesture_numpy[val_index]
    N_train,_,_,T,_=train_data.shape
    N_test,_,_,T,_=test_data.shape
    N_val,_,_,T,_=val_data.shape
    # (2047, 3, 114, 1800, 2)
    # print(train_data.shape)
    # exit(0)
    # train_data=train_data.reshape(N_train,T,-1)
    # test_data=test_data.reshape(N_test,T,-1)
    # val_data=val_data.reshape(N_val,T,-1)
    print(f"Train data shape: {train_data.shape}, Train label shape: {train_label.shape}")
    print(f"Val data shape: {val_data.shape}, Val label shape: {val_label.shape}")
    print(f"Test data shape: {test_data.shape}, Test label shape: {test_label.shape}")

    return train_data,train_label,test_data,test_label,val_data,val_label


def load_csida():
    train_X,train_y,test_X,test_y,val_X,val_y = extract_data_from_caida()
     #归一化步骤 
    # scaler = StandardScaler()
    # scaler.fit(train_X.reshape(-1, train_X.shape[-1])) #计算每一列的均值 
    # train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    # test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    # val_X = scaler.transform(val_X.reshape(-1, val_X.shape[-1])).reshape(val_X.shape)
    
    #处理label的部分
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y) #所有的label对应的类别列表 
    val_y = np.vectorize(transform.get)(val_y) #所有的label对应的类别列表 
    return train_X, train_y, test_X, test_y,val_X,val_y




class SSLneck(nn.Module):
    """Standard projection head: fc-bn-relu-fc-bn."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class ContrastiveHead(nn.Module):
    """Contrastive learning head (SimCLR/MoCo-style).

    Args:
        temperature (float): Temperature scaling factor for similarity scores.
    """

    def __init__(self, total_epochs=80, temperature_start=0.2 ,temperature_end = 0.07):
        super().__init__()
        self.tau_start = temperature_start
        self.tau_end = temperature_end
        self.total_epochs = total_epochs
        self.criterion = nn.CrossEntropyLoss()

    def get_temperature(self, epoch: int) -> float:
        """Linearly decays temperature from tau_start to tau_end."""
        tau = self.tau_start - (epoch / self.total_epochs) * (self.tau_start - self.tau_end)
        return tau

    def forward(self, pos: torch.Tensor, neg: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Args:
            pos (Tensor): [N, 1] positive similarities.
            neg (Tensor): [N, k] negative similarities.
            epoch (int): Current training epoch.

        Returns:
            Tensor: Scalar contrastive loss.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)  # [N, 1+k]
        tau = self.get_temperature(epoch)
        logits = logits / tau     # Apply temperature scaling

        labels = torch.zeros(N, device=pos.device, dtype=torch.long)

        loss = self.criterion(logits, labels)
        return loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        labels = torch.arange(batch_size).repeat(2)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)

        sim = sim / self.temperature
        sim_exp = torch.exp(sim)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
        sim_exp = sim_exp * (~mask)

        pos = sim_exp[labels.bool()].view(labels.shape[0], -1).sum(1)
        neg = sim_exp[~labels.bool()].view(labels.shape[0], -1).sum(1)

        loss = -torch.log(pos / (pos + neg)).mean()
        return loss

class ContrastiveTrainer:
    def __init__(self, args, device, logger,encoder,label_remap_dict, wandb=None):
        self.args = args
        self.device = device
        #self.writer = writer
        self.wandb = wandb
        self.logger = logger
        self.label_remap_dict = label_remap_dict
        self.encoder = encoder
        self.neck = SSLneck(input_dim=args.feat_dim, hidden_dim=128, output_dim=64).to(device) # feat dim这个值未定  zzy注:一个MLP
        self.head = ContrastiveHead(args.epochs,args.tau_start,args.tau_end)
        self.best_encoder=None

        self.best_probe_acc = 0.0
        self.best_encoder_weights = None
        self.best_probe_epoch = -1
    
    def _arc(self, x_anchor):
        rand_ant1,rand_ant2 = np.random.randint(low=0, high=x_anchor.size(1), size=2)
        rand_ant1,rand_ant2 = 0,1
        x_anchor_ant1 = x_anchor[:,rand_ant1,:,:,:].squeeze() # -> [batch, 2, sub, time]
        x_anchor_ant2 = x_anchor[:,rand_ant2,:,:,:].squeeze() # -> [batch, 2, sub, time]

        # zzy改的 CSIDA
        if x_anchor_ant1.shape[3]==2:
            # batchsize c t 2
            x_anchor_ant1=x_anchor_ant1.permute(0,3,1,2)
            x_anchor_ant2=x_anchor_ant2.permute(0,3,1,2)


        return x_anchor_ant1, x_anchor_ant2

    def _fda(self, x_anchor, x_env):
        """
        仅使用FDA作为预处理
        """
        if len(x_anchor.shape) == 5:  # x_anchor.shape: [batch, ant, 2, sub, time]
            # TODO 这里用到了的话需要修改
            # x_anchor shape [batch, ant, 2, sub, time]
            B, ant,_,sub, T = x_anchor.shape  
            x_anchor = x_anchor.reshape(x_anchor.size(0), -1, x_anchor.size(-1))  # -> [batch, ant*2*sub, time]
            x_env = x_env.reshape(x_env.size(0), -1, x_env.size(-1))  # -> [batch, ant*2*sub, time]
            x_i = x_anchor
            x_j = FDA_1d_with_fs(x_anchor, x_env, fs=1000, cutoff_freq=self.args.low, cutoff_freq_upper=self.args.high)
            x_i = x_i.view(B, ant, 2, sub, T)
            x_j = x_j.view(B, ant, 2, sub, T)
        else:  # x_anchor.shape: [batch, ant, sub,time] [b,2,224,224]
            x_anchor = x_anchor.reshape(x_anchor.size(0), -1, x_anchor.size(-1))  # -> [batch, 2*sub, time]
            x_env = x_env.reshape(x_env.size(0), -1, x_env.size(-1))  # -> [batch, 2*sub, time]
            x_i = x_anchor
            x_j = FDA_1d_with_fs(x_anchor, x_env, fs=1000, cutoff_freq=self.args.low, cutoff_freq_upper=self.args.high)
        return x_i, x_j

    def _arc_fda(self, x_anchor, x_env):
        rand_ant1,rand_ant2 = np.random.randint(low=0, high=x_anchor.size(1), size=2)
        rand_ant1,rand_ant2 = 0,1
        x_anchor_ant1 = x_anchor[:,rand_ant1,:,:,:].squeeze()
        x_anchor_ant2 = x_anchor[:,rand_ant2,:,:,:].squeeze()
        x_env_ant2 = x_env[:,rand_ant2,:,:,:].squeeze()
        # reshape -> [B, 2*S, T]
        B, C, S, T = x_anchor_ant1.shape  # C=2
        x_anchor_ant1 = x_anchor_ant1.reshape(B, -1, T)
        x_anchor_ant2 = x_anchor_ant2.reshape(B, -1, T)
        x_env_ant2    = x_env_ant2.reshape(B, -1, T)
        # FDA: 返回 [B, 2*S, T]
        x_i = x_anchor_ant1
        x_j = FDA_1d_with_fs(x_anchor_ant2, x_env_ant2, fs=1000, cutoff_freq=self.args.low, cutoff_freq_upper=self.args.high)
        # 恢复形状为 [B, 2, S, T]
        x_i = x_i.reshape(B, C, S, T)
        x_j = x_j.reshape(B, C, S, T)
        return x_i, x_j

    def _create_buffer(self,N):
        """Compute the mask and the index of positive samples.

        Args:
            N (int): batch size.
        """
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).to(self.device)
        pos_ind = (torch.arange(N * 2).to(self.device),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().to(self.device))

        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).to(self.device)
        neg_mask[pos_ind] = 0
        
        return mask, pos_ind, neg_mask

    def train(self, source_loader, target_loader,linear_train_loader,linear_eval_loader):
        """Train the model with self-supervised learning."""
        self.encoder.train()
        self.global_step = 0
        optimizer = torch.optim.Adam(
        list(self.encoder.parameters()) + list(self.neck.parameters()), lr=self.args.lr,weight_decay=self.args.weight_decay
    )
        # 定义warmup scheduler（前5个epoch线性上升）
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        # 定义cosine scheduler（剩下的 epoch）
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs - 5)
        # 组合调度器
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
        
        epoch_loss_min=100000
        for epoch in range(self.args.epochs):
            epoch_total_loss = 0.0
            source_loss_sum, target_loss_sum = 0, 0

            # --------- 第一阶段：对 source_loader 中的数据进行训练 ---------
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            target_batch = None

            for _ in range(len(source_loader)):
                source_batch = next(source_iter, None)
                if source_batch is None:
                    source_iter = iter(source_loader)
                    source_batch = next(source_iter)
                    target_batch = None

                loss = self._train_batch(source_batch, target_batch, optimizer,epoch)
                self.global_step += 1
                source_loss_sum += loss
                epoch_total_loss += loss

            # 更新学习率
            scheduler.step()
            # --------- 记录 ----------
            if self.args.fda_arc_choice != 'arc_src':
                avg_loss = epoch_total_loss / (len(source_loader) + len(target_loader))
                self.logger.nofmt(f"[SSL] Epoch {epoch}/{self.args.epochs}, Loss={avg_loss:.4f}")
                self.wandb.log({"ssl_train/loss_epoch": avg_loss, 
                            "ssl_train/loss_source_anchor": source_loss_sum / len(source_loader), 
                            "ssl_train/loss_target_anchor": target_loss_sum / len(target_loader)}, step=epoch)
            else:
                avg_loss = epoch_total_loss / (len(source_loader))
                self.logger.nofmt(f"[SSL] Epoch {epoch}/{self.args.epochs}, Loss={avg_loss:.4f}")
                self.wandb.log({"ssl_train/loss_epoch": avg_loss, 
                            "ssl_train/loss_source_anchor": source_loss_sum / len(source_loader)}, step=epoch)
            # Eval linear probe
            # is_best = self._linear_probe_eval(linear_train_loader,linear_eval_loader, epoch)
            is_best=False
            if epoch_loss_min>epoch_total_loss:
                # pass
                is_best=True
                epoch_loss_min=epoch_total_loss
                # self.best_encoder=self.encoder
                self.best_encoder = copy.deepcopy(self.encoder)
            # self.logger.nofmt(f"Best probe acc = {self.best_probe_acc:.4f} at epoch {self.best_probe_epoch}")
            # self.wandb.log({"ssl_eval/best_probe_acc": self.best_probe_acc}, step=epoch)
            self.logger.nofmt(f"loss  now is {epoch_total_loss}")
            self.logger.nofmt(f'loss min is {epoch_loss_min}')
            if is_best:
                self.best_encoder_weights = copy.deepcopy(self.encoder.state_dict())

        return self.encoder,self.best_encoder_weights  # 最后一轮 encoder、最佳 probe encoder

    def _forward_csiresnet(self,x):
        if len(x.shape) == 5:         # x.shape: [B, ant, 2, sub, time]
            # 原始 x: [B, A, 2, sub, time]
            B, A, C, S, T = x.shape
            # 将天线维度 A 合并进 batch 维度
            x_reshaped = x.permute(0,1,2,3,4).reshape(B * A, C * S, T)  # [B*A, 2*sub, time]
            # 一次性编码
            feat = self.encoder(x_reshaped)  # -> [B*A, D, T'] or [B*A, D] depends on encoder
            # 还原天线维度 A
            feat = feat.reshape(B, A, -1)  # [B, A, D]
            # 按天线特征拼接
            source_feat = feat.reshape(B, -1)  # [B, A*D]
        if len(x.shape) == 4: # x.shape: [B, 2, sub, time]
            # 原始 x: [B, 2, sub, time]
            B, C, S, T = x.shape
            # 将天线维度 A 合并进 batch 维度
            x_reshaped = x.reshape(B, -1, T)  # [B*A, 2*sub, time]
            # 一次性编码
            source_feat = self.encoder(x_reshaped)  # -> [B, D]
        return source_feat
    def _forward_resnet(self,x):
        if len(x.shape) == 5:         # x.shape: [B, ant, 2, sub, time]
            B, A, C, H, W = x.shape  # [B, ant, 2, sub, time]
            x_reshaped = x.view(B * A, C, H, W)   # -> [B * ant, 2, sub, time]
            feat_all = self.encoder(x_reshaped)  # -> [B * ant, D]
            source_feat = feat_all.view(B, A * feat_all.size(-1))  # -> [B, ant * D]
        else: # x shape = [B, 2, sub, time]
            source_feat = self.encoder(x)  # -> [B, D] 当前是[batch, 512]
        return source_feat

    def _train_batch(self, anchor_batch, env_batch, optimizer,epoch):
        # Move to device [batch, ant, sub, time, 2]
        x_anchor = anchor_batch[0].to(self.device) if isinstance(anchor_batch, (list, tuple)) else anchor_batch.to(self.device)
        if self.args.fda_arc_choice=='arc':
            x_env = env_batch[0].to(self.device) if isinstance(env_batch, (list, tuple)) else env_batch.to(self.device)
            x_concat = torch.cat([x_anchor, x_env], dim=0) # use both anchor and env for training
            x_i,x_j=self._arc(x_concat)
        elif self.args.fda_arc_choice=='fda':
            x_env = env_batch[0].to(self.device) if isinstance(env_batch, (list, tuple)) else env_batch.to(self.device)
            x_i,x_j=self._fda(x_anchor,x_env)
        elif self.args.fda_arc_choice=='arc_src':
            x_i,x_j=self._arc(x_anchor)
        else:
            x_env = env_batch[0].to(self.device) if isinstance(env_batch, (list, tuple)) else env_batch.to(self.device)
            x_i,x_j=self._arc_fda(x_anchor,x_env)
            
        # 编码 & 投影
        if self.args.backbone == 'CSIResNet':
            feat_i = self._forward_csiresnet(x_i)  # shape: [B, D]
            feat_j = self._forward_csiresnet(x_j)  # shape: [B, D]
        elif self.args.backbone == 'ResNet':
            feat_i = self._forward_resnet(x_i)
            feat_j = self._forward_resnet(x_j)
        z_i = self.neck(feat_i)
        z_j = self.neck(feat_j)
        z_i = z_i / (torch.norm(z_i, p=2, dim=1, keepdim=True) + 1e-10)
        z_j = z_j / (torch.norm(z_j, p=2, dim=1, keepdim=True) + 1e-10)

     
        z = torch.stack([z_i, z_j],dim=1) # (N,2,d)
  
        z = z.reshape((z.size(0) * 2,z.size(2)))


        N = z.size(0) // 2
        s = torch.matmul(z, z.T) # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # neg_mask , 2n*(2n-1)
        # remove diagonal, (2N)x(2N-1)

        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
 
        positive = s[pos_ind].unsqueeze(1) # (2N)x1
        
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        # Contrastive loss
        loss = self.head(positive, negative, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录
        #self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
        #self.wandb.log({"ssl_train/loss_step": loss.item()}, step=self.global_step)
        del x_i, x_j, feat_i, feat_j, z_i, z_j, s, positive, negative

        return loss.item()
    
    def _linear_probe_eval(self,train_loader,eval_loader,epoch):
        """"在 Linear Probe 的验证策略中，我们临时训练一个线性分类器（probe）,并立即评估其泛化能力。
        在这个过程中每次都随机初始化线性分类器的权重，并进行少量epoch的训练来评估特征质量"""
        self.encoder.eval()
        probe = Classifier(self.args.feat_dim,self.args.base_num_classes).to(self.device)
        optimizer_eval = torch.optim.Adam(probe.parameters(), lr=1e-3)

        for _ in range(5):  # Few epochs for probing
            for x, y in train_loader:
                x = x.to(self.device)
                mapped_y = torch.tensor([self.label_remap_dict[int(label)] for label in y], device=self.device)
                
                feat_list = []
                for ant in range(x.size(1)):
                    x_ant = x[:,ant,:,:,:].squeeze()  # -> [batch, 2, sub, time]
                    if x_ant.shape[3]==2:
                        x_ant=x_ant.permute(0,3,1,2)
                    #x_ant = x_ant.reshape(x_ant.size(0), -1, x_ant.size(-1))  # -> [batch, 2*ant*sub, time]
                    with torch.no_grad():
                        if self.args.backbone == 'CSIResNet':
                            feat = self._forward_csiresnet(x_ant)  # shape: [B, D]
                        elif self.args.backbone == 'ResNet':
                            feat = self._forward_resnet(x_ant)
                    feat_list.append(feat)
                feat = torch.sum(torch.stack(feat_list, dim=0), dim=0).squeeze()             
              
                logits = probe(feat)
                loss_eval = F.cross_entropy(logits, mapped_y)  # 使用函数形式的损失函数
                optimizer_eval.zero_grad()
                loss_eval.backward()
                optimizer_eval.step()
                
        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to(self.device)
                mapped_y = torch.tensor([self.label_remap_dict[int(label)] for label in y], device=self.device)
                feat_list = []
                for ant in range(x.size(1)):
                    x_ant = x[:,ant,:,:,:].squeeze()  # -> [batch, 2, sub, time]
                    if x_ant.shape[3]==2:
                        x_ant=x_ant.permute(0,3,1,2)
                    
                    #x_ant = x_ant.reshape(x_ant.size(0), -1, x_ant.size(-1))  # -> [batch, 2*ant*sub, time]
                    if self.args.backbone == 'CSIResNet':
                        feat = self._forward_csiresnet(x_ant)  # shape: [B, D]
                    elif self.args.backbone == 'ResNet':
                        feat = self._forward_resnet(x_ant)
                    feat_list.append(feat)
                feat = torch.sum(torch.stack(feat_list, dim=0), dim=0).squeeze()
                
                logits = probe(feat)
                pred = logits.argmax(dim=1)
                correct += (pred == mapped_y).sum().item()
                total += mapped_y.size(0)
        acc = correct / total
        #self.writer.add_scalar('ssl_eval/probe_acc', acc, epoch)
        self.logger.nofmt(f"[Probe Eval] Epoch {epoch}: Probe Acc={acc:.4f}, Correct={correct}/{total}")
        self.wandb.log({"ssl_eval/probe_acc": acc}, step=epoch)
        self.encoder.train()
        
        # 是否为最佳结果
        is_best = acc > self.best_probe_acc
        if is_best:
            self.best_probe_acc = acc
            self.best_probe_epoch = epoch
        return is_best


    def eval_classification_nonlinear(
        self, train_data, train_labels,
        test_data, test_labels,
        val_data, val_labels,
        batch_size, epochs, lr,
        hidden_size, num_classes
    ):
        """使用冻结的 best_encoder 进行非线性分类评估"""
        # 冻结特征提取器
        self.best_encoder.eval()
        diff = any((p1.data != p2.data).any() for p1, p2 in zip(self.encoder.parameters(), self.best_encoder.parameters()))
        print("❌ 不一样" if diff else "✅ 完全一样")
        for param in self.best_encoder.parameters():
            param.requires_grad = False

        # 构建数据集和 DataLoader
        batch_size=32
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float), torch.from_numpy(train_labels).to(torch.long))
        test_dataset  = TensorDataset(torch.from_numpy(test_data).to(torch.float), torch.from_numpy(test_labels).to(torch.long))
        val_dataset   = TensorDataset(torch.from_numpy(val_data).to(torch.float), torch.from_numpy(val_labels).to(torch.long))

        train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        test_loader  = DataLoader(test_dataset,  batch_size=min(batch_size, len(test_dataset)),  shuffle=False, drop_last=False)
        val_loader   = DataLoader(val_dataset,   batch_size=min(batch_size, len(val_dataset)),   shuffle=False, drop_last=False)

        # 初始化非线性分类器
        # model_classify = Classify_head(320, hidden_size, num_classes).to(self.device)
        # model_classify=nn.Linear(512,num_classes).to(device=self.device)
        model_classify = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ).to(device=self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_classify.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(epochs):
            model_classify.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                # fall_defi
                # b,a,c,t,x=data.shape
                # data=data.permute(0,1,4,2,3)

                # widar
                # data=data.permute(0,1,4,2,3)  # [b,a,c,t,x] -> [b,a,x,c,t]
                b,a,x,c,t=data.shape
                data= data.reshape(b,a,x*c,t)

                with torch.no_grad():
                    data_repr = self.best_encoder(data[:,0,:,:])+self.best_encoder(data[:,1,:,:])
                    # data_repr = self.best_encoder(data[:,0,:,:])+self.best_encoder(data[:,1,:,:])+self.best_encoder(data[:,2,:,:])
                    # data_repr = F.max_pool1d(data_repr.transpose(1, 2), kernel_size=data_repr.size(1)).transpose(1, 2)

                outputs = model_classify(data_repr.reshape(data_repr.size(0), -1))
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct_train / total_train
            print(f"[Train] Epoch {epoch}: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")

            # 验证
            model_classify.eval()
            correct_val, total_val = 0, 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(self.device), labels.to(self.device)

                    # falldefi
                    # b,a,c,t,x=data.shape
                    # data=data.permute(0,1,4,2,3)
                    # data= data.reshape(b,a,x*c,t)

                    b,a,x,c,t=data.shape
                    data= data.reshape(b,a,x*c,t)

                    data_repr = self.best_encoder(data[:,0,:,:])+self.best_encoder(data[:,1,:,:])

                    # data_repr = self.best_encoder(data[:,0,:,:])+self.best_encoder(data[:,1,:,:])+self.best_encoder(data[:,2,:,:])
                    outputs = model_classify(data_repr.reshape(data_repr.size(0), -1))
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_acc = 100 * correct_val / total_val
            scheduler.step(val_acc)
            print(f"[Val] Epoch {epoch}: Val Acc={val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model_classify.state_dict()

        # 加载最佳模型
        model_classify.load_state_dict(best_model_state)

        # 测试
        model_classify.eval()
        correct_test, total_test = 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # falldefi
                # b,a,c,t,x=data.shape
                # data=data.permute(0,1,4,2,3)
                # data= data.reshape(b,a,x*c,t)

                b,a,x,c,t=data.shape
                data= data.reshape(b,a,x*c,t)


                # data_repr = self.best_encoder(data)
                data_repr = self.best_encoder(data[:,0,:,:])+self.best_encoder(data[:,1,:,:])
                # data_repr = self.best_encoder(data[:,0,:,:])+self.best_encoder(data[:,1,:,:])+self.best_encoder(data[:,2,:,:])

                # data_repr = F.max_pool1d(data_repr.transpose(1, 2), kernel_size=data_repr.size(1)).transpose(1, 2)
                outputs = model_classify(data_repr.reshape(data_repr.size(0), -1))
                _, predicted = outputs.max(1)

                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        test_acc = 100 * correct_test / total_test
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        sensitivity_macro = recall_score(all_labels, all_preds, average='macro')

        print(f"[Test] Acc={test_acc:.2f}%, F1={f1_macro:.4f}, Sensitivity={sensitivity_macro:.4f}")
        return model_classify, test_acc, f1_macro, sensitivity_macro


        