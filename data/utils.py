import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms
from scipy.io import loadmat

from scipy.optimize import curve_fit
from numpy import unwrap
from scipy.signal import butter, filtfilt

from .transform import * #正常跑用
# from transform import * #debug用
# 先写一个 CSI基类，数据所有的数据处理完 都要通过这个变成dataset
class BaseCSIDataset(Dataset):
    def __init__(self, data, labels, domain=None,descriptation=None, transfrom=False):
        """
        Args:
            data (np.ndarray or torch.Tensor): CSI数据 (N, ...)
            labels (np.ndarray or torch.Tensor): 标签 (N,)
            domains_str (str): 一个字符串，列出有哪些源域，比如 'office lab outdoor'
            descriptation (str) 一个字符串 描述我这个set干了啥 
            transform (bool) 表明是否使用transformation
        """
        super(BaseCSIDataset, self).__init__()
        
        # 数据转换  
        self.data = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
        self.labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels
        self.descriptation=descriptation
        # 标签集合（升序去重）
        self.all_labels = torch.sort(self.labels.unique()).values.tolist()
        # 域处理
        self.domains=domain
        self.transform = transfrom
        # 检查
        assert len(self.data) == len(self.labels), "数据和标签长度不一致！"

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            sample = self.__transform__(sample)
        label = self.labels[index]
        return sample, label  # 这里只返回样本和标签 
    def __len__(self):
        return len(self.data)
    def get_domains(self):
        """返回当前域"""
        return self.domains
    def get_descriptation(self):
        return self.descriptation
    def get_all_labels(self):
        return self.all_labels  
    def __transform__(self,sample):
        # sample shape [ant,sub,time,amp+phase]
        transform_list = ['no_action', 'scaling_csi']
        return random_applied_trans(sample, transform_list, p=0.5, namespace=None)
        

class WiSDADataset(Dataset):
    def __init__(self, source_img_list, source_label_list):
        self.source_img_list = source_img_list
        self.source_label_list = source_label_list
        self.to_tensor = transforms.ToTensor()
        # self.transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])  # 通常用于ResNet预训练模型
# ])

    def __len__(self):
        return len(self.source_img_list)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.source_img_list[idx]
        image = Image.open(img_path).convert('RGB')  # 转成 RGB 格式
        image_np = np.array(image)

        # print(image.size)
        image = self.to_tensor(image)
        assert not torch.isnan(image).any(), f"Image contains NaN values at index {idx}, path: {img_path}"
        # image = self.transform(image)
        
        

        # print(image.shape)

        # 加载标签
        label = self.source_label_list[idx]

        # 转换为Tensor
        label = torch.tensor(label, dtype=torch.long)
        assert not torch.isnan(label).any(), f"Label is NaN at index {idx}, value: {label}"

        return image, label

class SignFiDataset(Dataset):
    def __init__(self, data, labels, domain=None,descriptation=None, transfrom=False,config=None):
        """
        Args:
            data (np.ndarray or torch.Tensor): CSI数据 (N, ...)
            labels (np.ndarray or torch.Tensor): 标签 (N,)
            domains_str (str): 一个字符串，列出有哪些源域，比如 'office lab outdoor'
            descriptation (str) 一个字符串 描述我这个set干了啥 
            transform (bool) 表明是否使用transformation
        """
        super(SignFiDataset, self).__init__()
        
        # 数据转换  
        # self.data = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
        self.data=data
        self.config=config
        #添加上归一化那？
        # self.data = (self.data - self.data.mean()) / self.data.std()
        self.labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels
        if self.labels.min() > 0:
            self.labels = self.labels - 1
        self.descriptation=descriptation
        # 标签集合（升序去重）
        self.all_labels = torch.sort(self.labels.unique()).values.tolist()

        # 域处理
        self.domains=domain
        self.transform = transfrom
        # 检查
        assert len(self.data) == len(self.labels), "数据和标签长度不一致！"

    def cal_ant_select_coeff(self,csi_data):
        """
            csi_data (np.ndarray): CSI data of shape [A, C, T] 
                                (Antennas, Subcarriers, Timesteps)
        """
        amp = np.abs(csi_data)  # [A, C, T]
        # print("amp type:", type(amp), "amp shape:", amp.shape)
        # Mean and variance over time axis
        mean_amp = np.mean(amp, axis=2)  # [A, C]
        var_amp = np.var(amp, axis=2)    # [A, C]
        # Stability score: mean / var (add epsilon to avoid division by zero)
        stability = mean_amp / (var_amp + 1e-8)  # [A, C]
        # Average over subcarriers
        stability_score = np.mean(stability, axis=1)  # [A]

        return stability_score


    def find_best_interval_by_phase_var(self,csiratio, sampling_rate=200, min_ms=5, max_ms=250, step_ms=5):
        """
        在[min_ms, max_ms]范围内枚举interval，找到使得相位差分后**方差最小**的interval

        Args:
            csiratio (np.ndarray): 复数CSI比值数据, shape = [C, T]
            sampling_rate (int): 采样率（Hz），默认 200Hz
            min_ms (int): 最小interval（单位ms）
            max_ms (int): 最大interval（单位ms）
            step_ms (int): 枚举步长（单位ms）

        Returns:
            best_interval (int): 最佳interval（单位ms）
            best_diff (np.ndarray): 最佳interval对应的差分结果，shape = [C, M]
        """
        assert csiratio.ndim == 2, "Input csiratio must be [C, T] shape"
        
        C, T = csiratio.shape
        phase = np.angle(csiratio)  # 转换为相位

        best_interval = None
        best_interval_pts=None
        best_diff = None
        min_phase_var = np.inf

        for interval_ms in range(min_ms, max_ms + 1, step_ms):
            interval_pts = int((interval_ms / 1000.0) * sampling_rate)
            if interval_pts <= 0 or T - interval_pts <= 0:
                continue  # 跳过非法interval

            # 相位差分
            diff = phase[:, interval_pts:] - phase[:, :-interval_pts]  # shape [C, T - interval]
            diff = np.unwrap(diff, axis=1)  # 解相位

            phase_var = np.var(diff)  # 所有子载波、所有时刻统一计算方差

            if phase_var < min_phase_var:
                min_phase_var = phase_var
                best_interval_pts=interval_pts
                best_interval = interval_ms
                best_diff = diff

        return best_interval_pts, best_diff

    
    def TD_computing_fixed(self,csiratio, interval_ms=80, sampling_rate=200):
        """
        在等间隔（均匀采样）情况下进行定间隔差分。
        
        Args:
            csiratio (np.ndarray): CSI 比值数据,shape = [C, T]
            interval_ms (float): 差分间隔，单位毫秒(默认80ms)
            sampling_rate (int): 采样率(Hz),默认200Hz

        Returns:
            np.ndarray: 差分结果，shape = [C, M]
        """
        C, T = csiratio.shape
        interval_pts = int((interval_ms / 1000.0) * sampling_rate)  # 转换成点数

        if interval_pts <= 0:
            raise ValueError("Interval too small for given sampling rate.")

        M = T - interval_pts  # 可用的差分数量
        if M <= 0:
            print("# 可用长度不足以做差分")
            return []

        # 直接做差分：csiratio[:, t+interval] - csiratio[:, t]
        diff = csiratio[:, interval_pts:] - csiratio[:, :-interval_pts]  # [C, M]
        return diff

    def normalize_csi_array(self,data: np.ndarray) -> np.ndarray:
        """
        对形状为 (2, C, T, 2) 的 CSI 数据进行归一化：
        - 第一个特征（最后一维的第0个）是振幅，归一化到 [0, π]
        - 第二个特征是相位，保持不变

        Args:
            data (np.ndarray): 输入的 CSI 数据，形状为 (2, C, T, 2)

        Returns:
            np.ndarray: 归一化后的 CSI 数据，形状同输入
        """
        assert data.shape[-1] == 2, "最后一个维度必须是振幅和相位"

        amp = data[..., 0]  # shape: [2, C, T]
        pha = data[..., 1]  # shape: [2, C, T]

        # 幅值归一化到 [0, π]
        amp_min = np.min(amp)
        amp_max = np.max(amp)
        amp_norm = (amp - amp_min) / (amp_max - amp_min + 1e-8)
        amp_norm = np.pi * amp_norm

        # 重组为 (2, C, T, 2)
        data_norm = np.stack([amp_norm, pha], axis=-1)

        return data_norm



    def remove_phase_offset_single_antenna(self,csi):
        """
        输入：
            csi: np.ndarray, shape (A, S, T), 复数 CSI，其中：
                A = 天线数量，S = 子载波数量，T = 时间帧数
        输出：
            csi_corrected: 相位偏移纠正后的复数 CSI，shape 同输入
        """
        
        A, S, T,_ = csi.shape
        # csi = np.array(csi)
        csi_abs = csi[:,:,:,0]
        csi_ang = csi[:,:,:,1]

        # 构建拟合自变量 x: [2, A*S]
        idx_tx_subc = np.zeros((2, A, S))
        for a in range(A):
            for s in range(S):
                idx_tx_subc[0, a, s] = (a + 2) / 3 - 2  # 天线编号（对齐旧代码）
                idx_tx_subc[1, a, s] = -58 + 4 * s      # 子载波编号
        idx_tx_subc = idx_tx_subc.reshape(2, -1)  # shape (2, A*S)

        # 获取第一个时间帧的相位并unwrap
        phase = csi_ang[..., 0]  # shape: (A, S)
        for a in range(A):
            phase[a, :] = unwrap(phase[a, :])  # 按子载波方向展开

        # flatten 为拟合目标值
        phase_flat = phase.flatten()

        # 定义拟合函数
        def func(x, a, b, c):
            return a * x[0] * x[1] + b * x[1] + c

        # 拟合系统性偏移模型
        popt, _ = curve_fit(func, idx_tx_subc, phase_flat)

        # 估计出来的系统性偏移，相同应用于所有时间帧
        phase_offset = func(idx_tx_subc, *popt).reshape(A, S)

        # 去除偏移
        for t in range(T):
            for a in range(A):
                csi_ang[a, :, t] = unwrap(csi_ang[a, :, t])
                csi_ang[a, :, t] -= phase_offset[a, :]

        # 重构复数 CSI
        # csi_corrected = csi_abs * np.exp(1j * csi_ang)
        csi_corrected=np.stack([csi_abs, csi_ang], axis=-1)
        def lowpass_filter(data, cutoff=70, fs=200, order=4):  # 假设采样率 1000Hz
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
            return filtfilt(b, a, data, axis=-1)

        csi_abs = lowpass_filter(csi_abs)     # <-- NEW
        csi_ang = lowpass_filter(csi_ang)     # <-- NEW
        # print(csi_corrected.shape)
        return csi_corrected


    def process_csi_sample_torch(self,sample):
        """
        输入:
            sample: torch.Tensor, shape (3, 30, 200, 2), 最后一维是 (amp, pha)
        输出:
            processed: torch.Tensor, shape (2, 30, 200, 2)
        """
        assert sample.shape == (3, 30, 200, 2), "输入shape必须是 (3,30,200,2)"

        amp = sample[..., 0]
        pha = sample[..., 1]
        # pha = np.unwrap(pha, axis=1)



        csi_complex = amp * np.exp(1j * pha)  # 复数张量
        stability_score = self.cal_ant_select_coeff(csi_complex)

        all_ants = {0, 1, 2}
        sel_ant1 = int(np.argmax(stability_score))
        ref_ant2 = int(np.argmin(stability_score))
        mid_ant3 = (all_ants - {sel_ant1, ref_ant2}).pop()
        csi_sel1 = csi_complex[sel_ant1,:,:].squeeze()
        csi_ref2 = csi_complex[ref_ant2,:,:].squeeze()
        csi_mid3 = csi_complex[mid_ant3,:,:].squeeze()
        # csi ratio
        csi_ratio1 = csi_sel1 / (csi_ref2 + 1e-8)       # [C, T]
        csi_ratio3 = csi_mid3 / (csi_ref2 + 1e-8)       # [C, T]

        min_csi_len = float('inf')
        max_csi_len = 0
        min_td_len = float('inf')
        max_td_len = 0
        current_len = csi_ratio1.shape[-1]
        min_csi_len = min(min_csi_len, current_len)
        max_csi_len = max(max_csi_len, current_len)


        interval_pts = int((80 / 1000.0) * 200)  # 转换成点数

        tdcsi1=self.TD_computing_fixed(csi_ratio1,80)
        # interv1,tdcsi1=self.find_best_interval_by_phase_var(csi_ratio1)

        tdcsi3=self.TD_computing_fixed(csi_ratio3,80)
        # interv3,tdcsi3=self.find_best_interval_by_phase_var(csi_ratio3)

        tdcsi1 = np.pad(tdcsi1, ((0, 0), (0, 16)), mode='constant', constant_values=0)  # (30, 200)
        tdcsi3 = np.pad(tdcsi3, ((0, 0), (0, 16)), mode='constant', constant_values=0)  # (30, 200)
        tdcsi = np.stack([tdcsi1, tdcsi3], axis=0)  # [2, C, T]
        amp_tdcsi = np.abs(tdcsi)   # [2, C, T]
        pha_tdcsi = np.angle(tdcsi)


        # 只做ratio
        # csi_ratio = np.stack([csi_ratio1, csi_ratio3], axis=0)  # [2, C, T]
        # amp=np.abs(csi_ratio)
        # pha=np.angle(csi_ratio)
        # combine_tdcsi = np.stack([amp, pha], axis=-1)



        combine_tdcsi = np.stack([amp_tdcsi, pha_tdcsi], axis=-1)

        # combine_tdcsi=self.normalize_csi_array(combine_tdcsi)

        # combine_tdcsi=self.normalize_csi_array(sample)
    
        return combine_tdcsi

    def __getitem__(self, index):
        sample = self.data[index]
        sample_process=self.remove_phase_offset_single_antenna(sample)
        # print(sample[:,:,:,0].shape)
        # sample_process=np.stack((sample[:,:,:,0],sample[:,:,:,1]),axis=-1)
        sample_tensor = torch.tensor(sample_process, dtype=torch.float32)

        if self.transform:
            sample = self.__transform__(sample)
        label = self.labels[index]
        return sample_tensor, label  # 这里只返回样本和标签 
    def __len__(self):
        return len(self.data)
    def get_domains(self):
        """返回当前域"""
        return self.domains
    def get_descriptation(self):
        return self.descriptation
    def get_all_labels(self):
        return self.all_labels  
    def __transform__(self,sample):
        # sample shape [ant,sub,time,amp+phase]
        transform_list = ['no_action', 'scaling_csi']
        return random_applied_trans(sample, transform_list, p=0.5, namespace=None)


class WidarDataset(Dataset):
    def __init__(self, file_path_list, label_list, transform=None, preload=False):
        self.file_path_list = file_path_list
        self.label_list = label_list
        self.transform = transform  # 如果你需要对 numpy 做 transform，可以传入 callable
        self.preload = preload

        if self.preload:
            print("⏫ 正在预加载数据到内存...")
            self.data = [np.load(p) for p in self.file_path_list]
            self.labels = [l for l in self.label_list]
            print("✅ 预加载完成")

        # 标签集合（升序去重）
        self.all_labels = torch.sort(torch.tensor(self.label_list).unique()).values.tolist()
    def get_all_labels(self):
        return self.all_labels  

    def pad_to_1000_timesteps(self,data):
        T, C1, C2, C3 = data.shape
        if T >= 500:
            return data[:500]  # 截断到 500
        # pad with zeros on time dimension
        pad_width = ((0, 500 - T), (0, 0), (0, 0), (0, 0))
        data_padded = np.pad(data, pad_width, mode='constant', constant_values=0)
        return data_padded

    def get_mat_data(self,file_path):
        #  等待完善，其实就是replace方法 替一下文件路径 把5个numpy数组读出来
        # file_path_data1=file_path

        file_name=os.path.basename(file_path)
        parent_path = os.path.dirname(file_path)          # /opt/data/private
        gra_path=os.path.dirname(parent_path)
        file_path_data1=os.path.join(gra_path,"h1divh3",file_name)
        file_path_data2=os.path.join(gra_path,"hivh3",file_name)
        file_path_data3=os.path.join(gra_path,"h1divh2",file_name)
        file_path_data4=os.path.join(gra_path,"h2",file_name)
        file_path_data5=os.path.join(gra_path,"h3",file_name)
        file_path_data6=os.path.join(gra_path,"h1",file_name)

        # 等待 替换路径 


        h1divh3 = loadmat(file_path_data1)['data1']
        h2divh3 =  loadmat(file_path_data2)['data2']
        h1divh2 =  loadmat(file_path_data3)['data3']
        h2 =  loadmat(file_path_data4)['data4']
        h3=  loadmat(file_path_data5)['data5']
        h1=  loadmat(file_path_data6)['data6']

        return h1divh3,h2divh3,h1divh2,h2,h3,h1

    # 等待完善 其实就是替换个 文件后缀
    def __len__(self):
        return len(self.file_path_list)
    

    def __getitem__(self, idx):

        is_mat=False
        if self.preload:
            data = self.data[idx]
            label = self.labels[idx]
        else:
            file_path=self.file_path_list[idx]
            file_name=os.path.basename(file_path)
            
            if file_name.endswith('.npy'):
                path = self.file_path_list[idx]
                data = np.load(path)
                label = self.label_list[idx]
            else:
                is_mat=True
                h1divh3,h2divh3,h1divh2,h2,h3,h1=self.get_mat_data(file_path) #这里的每个字如其名 形状都是(60,T) 30个子载波，前30是振幅 后30是相位 

                # TODO 需要写一个选择 函数 选择我们用什么数据 最后要合成一个data
                label=self.label_list[idx]
                


        if is_mat==False:
            if self.transform:
                data = self.transform(data)  # 对 numpy 做 transform，比如标准化等
            # 这里插值函数可能有问题要补0 
            if data.shape[0]!=500:
                data=self.pad_to_1000_timesteps(data)
            # 转换为 Tensor
            data = torch.from_numpy(data).float() #[t, sub, ant, 2]
            label = torch.tensor(label, dtype=torch.long)
            data = data.permute(2, 1, 0, 3) # [ant, sub, time, 2]
            return data, label
        else:
            # 等待完善 问题在于 形状不是1000了 应该用什么方法上下采样 ？？ maybe 上采样补0就ok 下的话 截取 or paa
            pass
            data = torch.from_numpy(data).float()
            label = torch.tensor(label, dtype=torch.long)
            return data,label

            # 
            
            




class WiGRUNT_dataset(Dataset):
    def __init__(self, source_img_list, source_label_list,transforme=True,preload=False):
        self.source_img_list = source_img_list
        self.source_label_list = source_label_list
        self.to_tensor = transforms.ToTensor()
        if transforme==True:
            self.transform=transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 指定目标尺寸，可改为你需要的大小
            transforms.ToTensor()
        ])
        self.preload=preload
        if self.preload:
            self.images = [Image.open(p).convert("RGB") for p in self.source_img_list]
            self.labels=[p for p in self.source_label_list]

    def __len__(self):
        return len(self.source_img_list)

    def __getitem__(self, idx):
        # 加载图像
        img_path=self.source_img_list[idx]
        if self.preload:
            image = self.images[idx]
            label=self.labels[idx]
        else:
            image = Image.open(self.source_img_list[idx]).convert("RGB")
            label = self.source_label_list[idx]

        # print(image.size)
        image = self.transform(image)
        # print(image.shape)

        # 加载标签
        basename=os.path.basename(img_path)
        ges_this=basename.split("_")[1]
        # if int(ges_this) != label:
        #     raise ValueError(f"标签有问题")


        # 转换为Tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# TODO maybe需要检查一下 label_list和读取的是否一样 以及dfs和图片读取是否正确
class Wiopen_dataset(Dataset):
    def __init__(self, source_img_list, source_label_list,transforme=True,preload=False):
        self.source_img_list = source_img_list
        
        
        self.source_label_list = source_label_list
        self.to_tensor = transforms.ToTensor()
        if transforme==True:
            self.transform=transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 指定目标尺寸，可改为你需要的大小
            transforms.ToTensor()
        ])
        self.preload=preload
        if self.preload:
            self.images = [Image.open(p).convert("RGB") for p in self.source_img_list]
            self.labels=[p for p in self.source_label_list]


    def __len__(self):
        return len(self.source_img_list)

    def __getitem__(self, idx):
        # 加载图像
        img_path=self.source_img_list[idx]
        if self.preload:
            image = self.images[idx]
            label=self.labels[idx]
        else:
            image = Image.open(self.source_img_list[idx]).convert("RGB")
            label = self.source_label_list[idx]
        # img_path_dfs = img_path.replace("data_CSIDA_Wiopen", "data_CSIDA_DFS")
        img_path_dfs = img_path.replace("data_widar_Wiopen", "data_widar_DFS")

        dfs=Image.open(img_path_dfs).convert("RGB")
        

        # print(image.size)
        image = self.transform(image)
        dfs=self.transform(dfs)
        # print(image.shape)

        # 加载标签
        basename=os.path.basename(img_path)
        ges_this=basename.split("_")[1]
        # if int(ges_this) != label:
        #     raise ValueError(f"标签有问题")


        # 转换为Tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, dfs,label,idx


def get_valid_csida_files(root_dir):
    files = os.listdir(root_dir)
    
    valid_prefixes = set()

    for file in files:
        if file.startswith('.'):
            continue  # 跳过隐藏文件

        if not file.endswith('.pkl'):
            continue  # 只处理pkl文件

        # 只要出现_CSIDA就切掉
        if '_CSIDA' in file:
            prefix = file.split('_CSIDA')[0]
            valid_prefixes.add(prefix)

    valid_prefixes = sorted(list(valid_prefixes))  # 排序，保持稳定

    return valid_prefixes

def select_data_with_class(data,label,dataset_name,a=150,select_set=None): #根据类别筛选出对应的data和label
    CSIDA_a=[0,2,3,5] 
    Widar_a=[]

    if dataset_name=='CSIDA':
        indices_in = np.where(np.isin(label, CSIDA_a))[0]
        indices_notin = np.where(~np.isin(label, CSIDA_a))[0]
        # indices = np.isin(label, CSIDA_a)
        # print(indices.shape)
        selected_data_in = data[indices_in]
        selected_label_in = label[indices_in]
        
        selected_data_not_in=data[indices_notin]
        selected_label_not_in=label[indices_notin]
        return selected_data_in, selected_label_in,selected_data_not_in,selected_label_not_in,CSIDA_a
        
        # with open('selected_label.txt', 'w') as f:
        #     for lab in selected_label:
        #         f.write(f"{lab}\n")
    elif dataset_name=='SignFi':
        if select_set==None:
            print(type(label))
            class_set = set(label.tolist())
            selected_set = random.sample(list(class_set), a)
            print(selected_set)
            print(type(selected_set))
            
            indices_in = np.where(np.isin(label, selected_set))[0]
            indices_notin = np.where(~np.isin(label, selected_set))[0]
            selected_data_in = data[indices_in]
            selected_label_in = label[indices_in]
            selected_data_not_in=data[indices_notin]
            selected_label_not_in=label[indices_notin]
            return selected_data_in, selected_label_in,selected_data_not_in,selected_label_not_in , selected_set #返回 选择的数据，以及选择的类别 方便target\
        else:
            # select_set=select_set
            indices_in = np.where(np.isin(label, select_set))[0]
            indices_notin = np.where(~np.isin(label, select_set))[0]
            selected_data_in = data[indices_in]
            selected_label_in = label[indices_in]
            selected_data_not_in=data[indices_notin]
            selected_label_not_in=label[indices_notin]
            return selected_data_in, selected_label_in,selected_data_not_in,selected_label_not_in , select_set    
    elif dataset_name=='Widar3.0':
        if select_set==None:
            print(type(label))
            class_set = set(label.tolist())
            selected_set=random.sample(list(class_set), a)
            indices_in = np.where(np.isin(label, selected_set))[0]
            indices_notin = np.where(~np.isin(label, selected_set))[0]
            selected_data_in = data[indices_in]
            selected_label_in = label[indices_in]
            selected_data_not_in=data[indices_notin]
            selected_label_not_in=label[indices_notin]
            return selected_data_in, selected_label_in,selected_data_not_in,selected_label_not_in , selected_set #返回 选择的数据，以及选择的类别 方便target\
        else:
            indices_in = np.where(np.isin(label, select_set))[0]
            indices_notin = np.where(~np.isin(label, select_set))[0]
            selected_data_in = data[indices_in]
            selected_label_in = label[indices_in]
            selected_data_not_in=data[indices_notin]
            selected_label_not_in=label[indices_notin]
            return selected_data_in, selected_label_in,selected_data_not_in,selected_label_not_in , select_set    
            
    else:
        raise ValueError("Invalid dataset name. Choose either 'CSIDA' or 'Widar'.") 
        

# class FAGes_d2a2dataset(Dataset):
#     def __init__(self, source_img_list, source_label_list,transforme=True,preload=True):
#         self.source_img_list = source_img_list
#         self.source_label_list = source_label_list
#         self.to_tensor = transforms.ToTensor()
#         if transforme==True:
#             self.transform=transform = transforms.Compose([
#             transforms.Resize((224, 224)),  # 指定目标尺寸，可改为你需要的大小
#             transforms.ToTensor()
#         ])
#         self.preload=preload
#         if self.preload:
#             self.images = []
#             for p in self.source_img_list:
#                 image_a1 = Image.open(p).convert("RGB").convert('L') 
#                 image_a1 = self.transform(image_a1)
#                 img_a2_path = p.replace("data_widar_wiopen", "data_widar_wiopen_h2h3")
#                 image_a2 = Image.open(img_a2_path).convert("RGB").convert('L')  
#                 image_a2 = self.transform(image_a2)
#                 self.images.append(torch.stack([image_a1, image_a2], dim=0).squeeze())  # shape: [2, 224, 224]
#             self.labels=[p for p in self.source_label_list]
#         # 标签集合（升序去重）
#         self.all_labels = torch.sort(torch.unique(torch.tensor(self.source_label_list))).values.tolist()

#     def get_all_labels(self):
#         return self.all_labels  
#     def __len__(self):
#         return len(self.source_img_list)

#     def __getitem__(self, idx):
#         # 加载图像
#         img_a1_path=self.source_img_list[idx]
#         img_a2_path = img_a1_path.replace("data_widar_wiopen", "data_widar_wiopen_h2h3")

#         if self.preload:
#             image_pair = self.images[idx]
#             label=self.labels[idx]
#         else:
#             image_a1 = Image.open(img_a1_path).convert("RGB")
#             image_a2 = Image.open(img_a2_path).convert("RGB")
#             label = self.source_label_list[idx]
#             image_a1 = image_a1.convert('L') 
#             image_a1= self.transform(image_a1) # [1,224,224]
#             image_a2 = image_a2.convert('L') 
#             image_a2= self.transform(image_a2)
#             image_pair = torch.stack([image_a1, image_a2], dim=0).squeeze()  # shape: [2, 224, 224]
#         label = torch.tensor(label, dtype=torch.long)
#         return image_pair, label

from collections import defaultdict
class FAGes_d2a2dataset(Dataset):
    def __init__(self, source_img_list, source_label_list, transforme=True, preload=False, ratio=None, mode='test'):
        self.source_img_list = source_img_list
        self.source_label_list = source_label_list
        self.preload = preload

        # 标签集合（升序去重）
        # self.all_labels = torch.sort(torch.unique(torch.tensor(self.source_label_list))).tolist()
        self.all_labels = torch.sort(torch.unique(torch.tensor(self.source_label_list))).values.tolist()


        # 根据 mode 和 ratio 决定是否进行按类采样
        if mode == 'train' and ratio is not None:
            # 将样本按类别分组
            label_to_samples = defaultdict(list)
            for img_path, label in zip(self.source_img_list, self.source_label_list):
                label_to_samples[label].append(img_path)

            # 按每类比例进行采样
            new_img_list = []
            new_label_list = []
            for label, samples in label_to_samples.items():
                k = max(1, int(len(samples) * ratio))  # 至少保留一个
                selected = random.sample(samples, k)
                # print(selected)
                new_img_list.extend(selected)
                new_label_list.extend([label] * k)

            self.source_img_list = new_img_list
            self.source_label_list = new_label_list

        # 加载图像数据
        if self.preload:
            self.images = [np.load(p) for p in self.source_img_list]
            self.labels = self.source_label_list  # 与 image_list 同步
        else:
            self.labels = self.source_label_list

    def get_all_labels(self):
        return self.all_labels  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.preload:
            image_pair = self.images[idx]
        else:
            image_pair = np.load(self.source_img_list[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_pair, label

class d2a2dataset(Dataset):
    def __init__(self, source_img_list, source_label_list,transforme=True,preload=False):
        self.source_img_list = source_img_list
        self.source_label_list = source_label_list
        self.to_tensor = transforms.ToTensor()
        if transforme==True:
            self.transform=transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 指定目标尺寸，可改为你需要的大小
            transforms.ToTensor()
        ])
        self.preload=preload
        if self.preload:
            self.images = [Image.open(p).convert("RGB") for p in self.source_img_list]
            self.labels=[p for p in self.source_label_list]
        # 标签集合（升序去重）
        self.all_labels = torch.sort(torch.unique(torch.tensor(self.source_label_list))).values.tolist()
    

    def get_all_labels(self):
        return self.all_labels  
    def __len__(self):
        return len(self.source_img_list)

    def __getitem__(self, idx):
        # 加载图像
        img_a1_path=self.source_img_list[idx]
        img_a2_path = img_a1_path.replace("data_widar_wiopen", "data_widar_wiopen_h2h3")

        if self.preload:
            image = self.images[idx]
            label=self.labels[idx]
        else:
            image_a1 = Image.open(img_a1_path).convert("RGB")
            image_a2 = Image.open(img_a2_path).convert("RGB")

            label = self.source_label_list[idx]

        # print(image.size)
        image_a1= self.transform(image_a1) # [3,224,224]
        image_a2= self.transform(image_a2)

        # 加载标签
        # basename=os.path.basename(img_path)
        # ges_this=basename.split("_")[1]
        # if int(ges_this) != label:
        #     raise ValueError(f"标签有问题")

        # 转换为Tensor
        label = torch.tensor(label, dtype=torch.long)

        return image_a1,image_a2, label

def select_data_with_k_shot(data, label, k):
    """
    根据类别筛选出每个类k个样本（做到：每个类别生成一个子字典）
    """
    print("label.shape:", label.shape)
    #print("label内容:", label)
    
    # 第一步：找到所有类别
    class_set = set(label.tolist())
    print("检测到的所有类别有:", class_set)
    print("检测到的所有类别数:", len(class_set))
    
    # 第二步：每个类别对应一个字典 {类别: [索引们]}
    class_indices_list = []

    for c in sorted(class_set):
        indices = np.where(label == c)[0]  # 找出属于类别c的样本索引
        indices = indices.tolist()
        class_indices_list.append({c: indices})  # 这里放到字典里面！
        # print(c, "类别的索引:", indices)

    #第三步,从每个类别中随机选择k个样本进行剥离  
    k_shot_indices = []      # 存放所有k-shot索引
    non_k_shot_indices = []  # 存放所有剩余的索引
    for item in class_indices_list:
        for cls, indices in item.items():
            
            if len(indices) < k:
                raise ValueError(f"类别{cls}的样本数量只有{len(indices)}个，少于k={k}，无法取样！")
            selected = random.sample(indices, k)  # 从indices中随机挑k个
            not_selected = list(set(indices) - set(selected))  # 剩下的

            print(f"类别{cls}: 选中的k-shot样本索引: {selected}")
            # print(f"类别{cls}: 剩余的non-k-shot样本索引: {not_selected}")

            k_shot_indices.extend(selected)
            non_k_shot_indices.extend(not_selected)
    
    #print(k_shot_indices)
    #print(non_k_shot_indices)
    data_suppot=data[k_shot_indices]
    label_suppot=label[k_shot_indices]
    data_query=data[non_k_shot_indices]
    label_query=label[non_k_shot_indices]
    # print(data_suppot.shape)
    # print(label_suppot.shape)
    # print(data_query.shape)
    # print(label_query.shape)
    return data_suppot, label_suppot, data_query, label_query

    
# root_dir = "/data0/zzy25/CSIDA/CSIDA/CSI_301/"
# files = get_valid_csida_files(root_dir)
# print(files)


# import numpy as np  
# res=np.load("/data0/zzy25/widar3/no_process_this/20181211_room_room3_Clap_user_user8_location_3_orientation_3_repetition_5.npy")
# print(res.shape)




class visual_dataset(Dataset):
    def __init__(self, source_img_list, source_label_list,transforme=True,preload=False):
        self.source_img_list = source_img_list
        self.source_label_list = source_label_list
        self.to_tensor = transforms.ToTensor()
        if transforme==True:
            self.transform= transforms.Compose([
            transforms.Resize((224, 224)),  # 指定目标尺寸，可改为你需要的大小
            transforms.ToTensor()
        ])
        self.preload=preload
        if self.preload:
            self.images = [Image.open(p).convert("RGB") for p in self.source_img_list]
            self.labels=[p for p in self.source_label_list]

    def __len__(self):
        return len(self.source_img_list)

    def __getitem__(self, idx):
        # 加载图像
        img_path=self.source_img_list[idx]
        if self.preload:
            image = self.images[idx]
            label=self.labels[idx]
        else:
            image = Image.open(self.source_img_list[idx]).convert("RGB")
            label = self.source_label_list[idx]

        # print(image.size)
        image = self.transform(image)
        # print(image.shape)

        # 加载标签
        basename=os.path.basename(img_path)
        ges_this=basename.split("_")[1]
        # if int(ges_this) != label:
        #     raise ValueError(f"标签有问题")


        # 转换为Tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label,img_path


class visual_dataset_1d(Dataset):
    def __init__(self, source_img_list, source_label_list,transforme=True,preload=False):
        self.source_img_list = source_img_list
        self.source_label_list = source_label_list
        self.to_tensor = transforms.ToTensor()
        if transforme==True:
            self.transform=transforms.Compose([
            transforms.Resize((224, 224)),  # 指定目标尺寸，可改为你需要的大小
            transforms.ToTensor()
        ])
        self.preload=preload
        if self.preload:
            self.images = [Image.open(p).convert("RGB") for p in self.source_img_list]
            self.labels=[p for p in self.source_label_list]

    def __len__(self):
        return len(self.source_img_list)

    def __getitem__(self, idx):
        # 加载图像
        path=self.source_img_list[idx]
        if self.preload:
            image = self.images[idx]
            label=self.labels[idx]
        else:
            data = np.load(path)
            label = self.source_label_list[idx]

        data = torch.from_numpy(data).float()
        # 转换为Tensor
        label = torch.tensor(label, dtype=torch.long)

        return data, label,path



# TODO 分domain 分domain需要self.config去做 

class DGDataset(Dataset):
    def __init__(self, file_path_list, label_list, transform=None, preload=False, config=None):
        """
        file_path_list: list of str, 每个样本的.npy路径
        label_list: list of int, 每个样本的标签
        transform: callable, 可选，对数据做预处理
        preload: bool, 如果True，预加载数据到内存
        config: 配置对象
        """
        self.file_path_list = file_path_list
        self.label_list = label_list
        self.transform = transform
        self.preload = preload
        self.config = config

        # 处理 domain，得到 domain_label 和 domain_counts
        self.domain_label, self.domain_counts = self.parse_domains()
        self.domain_indices = self.build_domain_indices()
        self.label_idx_map = self.get_label_map()
        # print(self.label_idx_map[])
   


        if self.preload:
            print("⏫ 正在预加载数据到内存...")
            self.data = [np.load(p) for p in self.file_path_list]
            print("✅ 预加载完成")

    def parse_domains(self):
        """
        从路径中提取 room, location, orientation 拼接成 domain 字符串。
        返回:
        - domain_label: list of int，重映射后的域编号
        - domain_counts: dict, 每个域编号的样本数
        """
        raw_domains = []
        # /opt/data/private/ablation_study/data_widar_800/data_tdcsi_dfs/20181109_room_room1_Clap_user_user1_location_1_orientation_1_repetition_1.png

        for file_path in self.file_path_list:
            file_name = os.path.basename(file_path)
            parts = file_name.replace(".npy", "").split("_")

            # 提取 room
            if "room" in parts:
                room = parts[parts.index("room") + 1]
            else:
                raise ValueError(f"'room' not found in file name: {file_name}")

            # 提取 location
            if "location" in parts:
                location = parts[parts.index("location") + 1]
            else:
                raise ValueError(f"'location' not found in file name: {file_name}")

            # 提取 orientation
            if "orientation" in parts:
                orientation = parts[parts.index("orientation") + 1]
            else:
                raise ValueError(f"'orientation' not found in file name: {file_name}")

            # 拼接成 domain 字符串
            domain = f"{room}_location_{location}_orientation_{orientation}"
            raw_domains.append(domain)

        # 🌟 重映射 domain 到连续整数编号
        unique_domains = sorted(set(raw_domains))
        domain_remap = {domain: i for i, domain in enumerate(unique_domains)}
        self.domainremap_re={i:domain  for i, domain in enumerate(unique_domains)}
        


        domain_label = []
        domain_counts = {i: 0 for i in range(len(unique_domains))}

        for domain in raw_domains:
            mapped = domain_remap[domain]
            domain_label.append(mapped)
            domain_counts[mapped] += 1

        return domain_label, domain_counts

    def build_domain_indices(self):
        """
        构建 domain_indices: 每个域编号对应的索引列表
        返回:
        - dict: {domain_id: [indices]}
        """
        domain_indices = {}
        for idx, domain in enumerate(self.domain_label):
            if domain not in domain_indices:
                domain_indices[domain] = []
            domain_indices[domain].append(idx)
        return domain_indices

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        if self.preload:
            data = self.data[idx]
        else:
            data = np.load(self.file_path_list[idx])

        label = self.label_list[idx]
        domain = self.domain_label[idx]

        if self.transform:
            data = self.transform(data)

        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)
        domain = torch.tensor(domain, dtype=torch.long)

        return data, label, domain, self.domainremap_re[domain.item()],idx




class LiSADataset(Dataset):
    def __init__(self, file_path_list, label_list, transform=None, preload=False, config=None):
        """
        file_path_list: list of str, 每个样本的.npy路径
        label_list: list of int, 每个样本的标签
        transform: callable, 可选，对数据做预处理
        preload: bool, 如果True，预加载数据到内存
        config: 配置对象
        """
        self.file_path_list = file_path_list
        self.label_list = label_list
        self.transform = transform
        self.preload = preload
        self.config = config

        # 处理 domain，得到 domain_label 和 domain_counts
        self.domain_label, self.domain_counts = self.parse_domains()
        self.domain_indices = self.build_domain_indices()
        self.label_idx_map = self.get_label_map()
        
        if self.preload:
            print("⏫ 正在预加载数据到内存...")
            self.data = [np.load(p) for p in self.file_path_list]
            print("✅ 预加载完成")
        
        # 标签集合（升序去重）
        self.all_labels = torch.sort(torch.unique(torch.tensor(self.label_list))).values.tolist()
        # 域标签集合（升序去重）
        self.all_domain_labels = torch.sort(torch.unique(torch.tensor(self.domain_label))).values.tolist()

        # TODO 针对半监督的修改    
        self.labeled_ratio  = config.ratio
        self.labeled_indices = []
        self.unlabeled_indices = []
        if self.labeled_ratio is not None:
            self._split_labeled_unlabeled()
            self.label_flags = [False] * len(self.file_path_list)
            for idx in self.labeled_indices:
                self.label_flags[idx] = True
        else:
            self.label_flags = [True] * len(self.file_path_list)  # 全部有标签

    def _split_labeled_unlabeled(self):
        """
        根据 labeled_ratio 划分有标签和无标签样本索引。
        """
        all_indices = set(range(len(self.file_path_list)))
        labeled_indices_list = []
        
        for label in self.all_labels:
            indices_for_label = self.label_idx_map[label]
            # 根据比例计算有标签样本数，并取整
            num_labeled = int(len(indices_for_label) * self.labeled_ratio)
            
            # 确保至少有一个有标签样本（如果比例大于0）
            if self.labeled_ratio > 0 and num_labeled == 0:
                num_labeled = 1

            if len(indices_for_label) < num_labeled:
                print(f"警告：类别 {label} 的样本数 ({len(indices_for_label)}) 小于 {num_labeled}。将使用所有样本作为有标签数据。")
                labeled_indices_list.extend(indices_for_label)
            else:
                labeled_indices_list.extend(random.sample(indices_for_label, num_labeled))

        self.labeled_indices = labeled_indices_list
        self.unlabeled_indices = list(all_indices - set(self.labeled_indices))
        print(f"✅ 数据集已根据比例划分：总样本数={len(all_indices)}, 有标签样本数={len(self.labeled_indices)}")

    def parse_domains(self):
        """
        从路径中提取 room, location, orientation 拼接成 domain 字符串。 
        返回:
        - domain_label: list of int，重映射后的域编号
        - domain_counts: dict, 每个域编号的样本数
        """
        raw_domains = []

        for file_path in self.file_path_list:
            file_name = os.path.basename(file_path)
            parts = file_name.replace(".npy", "").split("_")

            # 提取 room
            if 'room' in parts:
                raw_room = parts[parts.index('room') + 1]
                room_num = raw_room.replace('room', '')
                room = f"room_{room_num}"
            else:
                raise ValueError(f"'room' not found in file name: {file_name}")

            # 提取 user，保留前缀
            if 'user' in parts:
                raw_user = parts[parts.index('user') + 1]
                user_num = raw_user.replace('user', '')
                user = f"user_{user_num}"
            else:
                raise ValueError(f"'user' not found in file name: {file_name}")

            # 提取 location
            if 'location' in parts:
                location = parts[parts.index('location') + 1]
            else:
                raise ValueError(f"'location' not found in file name: {file_name}")

            # 提取 orientation
            if 'orientation' in parts:
                orientation = parts[parts.index('orientation') + 1]
            else:
                raise ValueError(f"'orientation' not found in file name: {file_name}")

            # 拼接成 domain 字符串
            domain = f"{room}_{user}_location_{location}_orientation_{orientation}"

            raw_domains.append(domain)

        # 🌟 重映射 domain 到连续整数编号
        unique_domains = sorted(set(raw_domains))
        domain_remap = {domain: i for i, domain in enumerate(unique_domains)}
        self.domain_remap=domain_remap
        self.domainremap_re={i:domain  for i, domain in enumerate(unique_domains)}

        domain_label = []
        domain_counts = {i: 0 for i in range(len(unique_domains))}
  

        for domain in raw_domains:
            mapped = domain_remap[domain]
            domain_label.append(mapped)
            domain_counts[mapped] += 1

        return domain_label, domain_counts


    def build_domain_indices(self):
        """
        构建 domain_indices: 每个域编号对应的索引列表
        返回:
        - dict: {domain_id: [indices]}
        """
        domain_indices = {}
        for idx, domain in enumerate(self.domain_label):
            if domain not in domain_indices:
                domain_indices[domain] = []
            domain_indices[domain].append(idx)
        return domain_indices


    def get_label_map(self):
        """
        构建 label_map:key 是类别标签，value 是该标签对应的样本索引集合。
        返回:
            dict: { label: set(idx1, idx2, ...) }
        """
        label_map = {}
        for idx, label in enumerate(self.label_list):
            label = int(label)
            if label not in label_map:
                label_map[label] = []
            label_map[label].append(idx)

        return label_map

    def get_sample_different_domain_simple(self,y, domain, idx):
        """
        简单高效版本：
        随机选样本索引，如果 domain 不同直接返回。
        最多尝试 30 次，找不到就返回原样本。
        """
        max_attempts = 30
        for _ in range(max_attempts):
            idx_choose = np.random.randint(len(self.file_path_list))
            if self.domain_label[idx_choose] != domain.item():
                data = torch.from_numpy(np.load(self.file_path_list[idx_choose])).float()
                label = torch.tensor(self.label_list[idx_choose])
                domain_label = torch.tensor(self.domain_label[idx_choose])
                return data, label, domain_label, idx_choose

        # 找不到，返回原样本
        data = torch.from_numpy(np.load(self.file_path_list[idx.item()])).float()
        label = torch.tensor(self.label_list[idx.item()])
        domain_label = torch.tensor(self.domain_label[idx.item()])
        return data, label, domain_label, idx.item()

    def get_sample_different_domain_same_label_simple(self, y, domain, idx):
        """
        根据当前标签，在同类样本中随机找 domain 不同的样本。
        最多尝试 30 次，找不到就返回随机样本。
        """
        max_attempts = 30
        label = y.item()
        # 获取该标签对应的所有 idx 集合，并转成 list 以便随机取
        candidate_indices = self.label_idx_map[label]
        for _ in range(max_attempts):
            idx_choose = np.random.choice(candidate_indices)
            domain_choose = self.domain_label[idx_choose]
            if domain_choose != domain.item():
                data = torch.from_numpy(np.load(self.file_path_list[idx_choose])).float()
                label_tensor = torch.tensor(self.label_list[idx_choose])
                domain_label_tensor = torch.tensor(domain_choose)
                return data, label_tensor, domain_label_tensor, idx_choose
        # 如果30次都没找到，随便返回一个
        idx_choose = np.random.randint(len(self.file_path_list))
        data = torch.from_numpy(np.load(self.file_path_list[idx_choose])).float()
        label_tensor = torch.tensor(self.label_list[idx_choose])
        domain_label_tensor = torch.tensor(self.domain_label[idx_choose])
        return data, label_tensor, domain_label_tensor, idx_choose

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        if self.preload:
            data = self.data[idx]
        else:
            data = np.load(self.file_path_list[idx])

        label = self.label_list[idx]
        domain = self.domain_label[idx]

        if self.transform:
            data = self.transform(data)

        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)
        domain = torch.tensor(domain, dtype=torch.long)

        # 判断是否为有标签数据
        is_labeled = self.label_flags[idx]
        # 对于有标签数据返回标签，否则返回-1
        if is_labeled:
            return data, label, domain, self.domainremap_re[domain.item()], idx
        else: # 因为label是从0开始，直接乘-1会导致0变成0，所以乘-1再减1
            return data, -1*label-1, domain, self.domainremap_re[domain.item()], idx
    
    def get_all_labels(self):
        return self.all_labels  
    def get_all_domain_labels(self):
        return self.all_domain_labels

class LiSADataset_CSIDA(Dataset):
    def __init__(self, data, label, domain=None,transform=None, preload=False, config=None):
        """
        file_path_list: list of str, 每个样本的.npy路径
        label_list: list of int, 每个样本的标签
        transform: callable, 可选，对数据做预处理
        preload: bool, 如果True，预加载数据到内存
        config: 配置对象
        """
        # self.file_path_list = file_path_list
        self.data=data
        self.label = label
        self.domain=domain
        domain_to_id = {domain_str: idx for idx, domain_str in enumerate(sorted(set(self.domain)))}
        self.domain_label = [domain_to_id[d] for d in self.domain]

        self.transform = transform
        self.preload = preload
        self.config = config
        self.label_idx_map = self.get_label_map()

        # 标签集合（升序去重）
        self.all_labels = torch.sort(torch.unique(torch.tensor(self.label))).values.tolist()
        # 域标签集合（升序去重）
        self.all_domain_labels = torch.sort(torch.unique(torch.tensor(self.domain_label))).values.tolist()    

    def get_label_map(self):
        """
        构建 label_map:
        key 是类别标签，value 是该标签对应的样本索引集合。
        
        返回:
            dict: { label: set(idx1, idx2, ...) }
        """
        label_map = {}

        for idx, label in enumerate(self.label):
            label = int(label)
            if label not in label_map:
                label_map[label] = []
            label_map[label].append(idx)

        return label_map

    def get_sample_different_domain_same_label_simple(self, y, domain, idx):
        """
        根据当前标签，在同类样本中随机找 domain 不同的样本。
        最多尝试 30 次，找不到就返回随机样本。
        """
        max_attempts = 30
        label = y.item()

        # 获取该标签对应的所有 idx 集合，并转成 list 以便随机取
        candidate_indices = self.label_idx_map[label]

        for _ in range(max_attempts):
            idx_choose = np.random.choice(candidate_indices)
            domain_choose = self.domain_label[idx_choose]

            if domain_choose != domain.item():
                data = torch.from_numpy(self.data[idx_choose]).float()
                label_tensor = torch.tensor(self.label[idx_choose])
                domain_label_tensor = torch.tensor(domain_choose)
                return data, label_tensor, domain_label_tensor, idx_choose

        # 如果30次都没找到，随便返回一个
        idx_choose = np.random.randint(len(self.data))
        data = torch.from_numpy(self.data[idx_choose]).float()
        label_tensor = torch.tensor(self.label[idx_choose])
        domain_label_tensor = torch.tensor(self.domain_label[idx_choose])
        return data, label_tensor, domain_label_tensor, idx_choose


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data=self.data[idx]
        label = self.label[idx]
        domain = self.domain_label[idx]

        if self.transform:
            data = self.transform(data)

        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)
        domain = torch.tensor(domain, dtype=torch.long)

        return data, label, domain, domain,idx

    def get_all_labels(self):
        return self.all_labels  
    def get_all_domain_labels(self):
        return self.all_domain_labels


class DARCDataset(Dataset):
    def __init__(self, file_path_list, label_list, cross_domain_type, transform=None, preload=False, config=None):
        """
        file_path_list: list of str, 每个样本的.npy路径
        label_list: list of int, 每个样本的标签
        transform: callable, 可选，对数据做预处理
        preload: bool, 如果True，预加载数据到内存
        config: 配置对象
        cross_domain_type: str, 指定跨域类型（room, user, location
        """
        self.file_path_list = file_path_list
        self.label_list = label_list
        self.transform = transform
        self.preload = preload
        self.config = config
        self.cross_domain_type = cross_domain_type

        # 处理 domain，得到 domain_label 和 domain_counts
        self.domain_label, self.domain_counts = self.parse_domains()
        self.domain_indices = self.build_domain_indices()
        self.label_idx_map = self.get_label_map()
        
        if self.preload:
            print("⏫ 正在预加载数据到内存...")
            self.data = [np.load(p) for p in self.file_path_list]
            print("✅ 预加载完成")
        
        # 标签集合（升序去重）
        self.all_labels = torch.sort(torch.unique(torch.tensor(self.label_list))).values.tolist()
        # 域标签集合（升序去重）
        self.all_domain_labels = torch.sort(torch.unique(torch.tensor(self.domain_label))).values.tolist()


    def parse_domains(self):
        """
        从路径中提取 room, location, orientation 拼接成 domain 字符串。 
        返回:
        - domain_label: list of int，重映射后的域编号
        - domain_counts: dict, 每个域编号的样本数
        """
        raw_domains = []

        for file_path in self.file_path_list:
            file_name = os.path.basename(file_path)
            parts = file_name.replace(".npy", "").split("_")

            # 提取 room
            if 'room' in parts:
                raw_room = parts[parts.index('room') + 1]
                room_num = raw_room.replace('room', '')
                room = f"room_{room_num}"
            else:
                raise ValueError(f"'room' not found in file name: {file_name}")

            # 提取 user，保留前缀
            if 'user' in parts:
                raw_user = parts[parts.index('user') + 1]
                user_num = raw_user.replace('user', '')
                user = f"user_{user_num}"
            else:
                raise ValueError(f"'user' not found in file name: {file_name}")

            # 提取 location
            if 'location' in parts:
                location = parts[parts.index('location') + 1]
            else:
                raise ValueError(f"'location' not found in file name: {file_name}")

            # 提取 orientation
            if 'orientation' in parts:
                orientation = parts[parts.index('orientation') + 1]
            else:
                raise ValueError(f"'orientation' not found in file name: {file_name}")

            # 🔑 根据 cross_domain_type 选择 domain
            if self.cross_domain_type == "room":
                domain = room
            elif self.cross_domain_type == "user":
                domain = user
            elif self.cross_domain_type == "location":
                domain = location
            elif self.cross_domain_type == "orientation":
                domain = orientation
            else:
                raise ValueError(f"Invalid cross_domain_type: {self.cross_domain_type}")

            raw_domains.append(domain)

        # 🌟 重映射 domain → 连续整数编号
        unique_domains = sorted(set(raw_domains))
        domain_remap = {domain: i for i, domain in enumerate(unique_domains)}
        self.domain_remap = domain_remap
        self.domainremap_re = {i: domain for i, domain in enumerate(unique_domains)}

        domain_label = []
        domain_counts = {i: 0 for i in range(len(unique_domains))}
  

        for domain in raw_domains:
            mapped = domain_remap[domain]
            domain_label.append(mapped)
            domain_counts[mapped] += 1

        return domain_label, domain_counts


    def build_domain_indices(self):
        """
        收集每个域的所有样本
        构建 domain_indices: 每个域编号对应的索引列表
        返回:
        - dict: {domain_id: [indices]}
        """
        domain_indices = {}
        for idx, domain in enumerate(self.domain_label):
            if domain not in domain_indices:
                domain_indices[domain] = []
            domain_indices[domain].append(idx)
        return domain_indices

    def get_label_map(self):
        """
        将所有的样本按标签分组
        构建 label_map:
        key 是类别标签，value 是该标签对应的样本索引集合。
        
        返回:
            dict: { label: set(idx1, idx2, ...) }
        """
        label_map = {}

        for idx, label in enumerate(self.label_list):
            label = int(label)
            if label not in label_map:
                label_map[label] = []
            label_map[label].append(idx)

        return label_map

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        if self.preload:
            data = self.data[idx]
        else:
            data = np.load(self.file_path_list[idx])

        label = self.label_list[idx]
        domain = self.domain_label[idx]

        if self.transform:
            data = self.transform(data)

        data = torch.from_numpy(data).float()
        
        label = torch.tensor(label, dtype=torch.long)
        domain = torch.tensor(domain, dtype=torch.long)

        return data, label, domain, self.domainremap_re[domain.item()],idx
    
    def get_all_labels(self):
        return self.all_labels  
    def get_all_domain_labels(self):
        return self.all_domain_labels