import numpy as np
import re,os,pickle
import scipy.io as scio
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import scipy.signal as signal
# import pywt
import csv
import yaml
import time
from sklearn.decomposition import PCA
import json
# from torch.autograd.profiler import profile, record_function, ProfilerActivity
from sklearn.cluster import KMeans
from scipy.signal.windows import gaussian


def phase_deg_to_complex(amp, phase_deg):
    phase_rad = np.deg2rad(phase_deg)  # 把度转弧度
    z = amp * np.exp(1j * phase_rad)   # 还原复数
    return z

def guess_phase_unit(phase_arr):
    """
    简单判断相位数组是弧度还是度。
    规则：
    - 弧度通常范围在 [-2π, 2π] (~ -6.28 ~ 6.28)左右
    - 度通常范围在 [-360, 360]左右
    - 根据最大绝对值判断，更接近哪个范围则判断为该单位
    """
    max_abs = np.max(np.abs(phase_arr))
    print(max_abs)
    if max_abs <= 2 * np.pi:
        return "radians"
    elif max_abs <= 360:
        return "degrees"
    else:
        # 数值异常，超出常规范围，可能是展开相位或者其他
        return "unknown (possibly unwrapped phase or unusual range)"

def read_CSIDA(amp_file_path,phase_filepath,label_file_path):
    # pass
    # print(os.path.basename(amp_file_path))
    base_name=os.path.basename(amp_file_path)
    split_name=base_name.split("_")
    # print(split_name)
    with open(amp_file_path, 'rb') as f:
        amp_data = pickle.load(f)
        # print(amp_data.shape)
    with open(label_file_path,'rb') as f:
        label_data = pickle.load(f)
        # print(label_data.shape)
    with open(phase_filepath, 'rb') as f:
        pha_data = pickle.load(f)
        # print(pha_data.shape)
        # print(set(label_data))
        # print(amp_data.shape)
    return amp_data,pha_data,label_data




def read_CSIDA_file_path(filepath):
    """"读取CSIDA数据集数据地址"""
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if not file.startswith(".") and file.endswith("amp.pkl"):
                amp_filepath = os.path.join(root, file)
                label_filepath=amp_filepath.replace("amp.pkl","label.pkl") 
                phase_filepath=amp_filepath.replace("amp.pkl","pha.pkl") 

                amp,pha,label=read_CSIDA(amp_filepath,phase_filepath,label_filepath)
                return amp,pha,label
# https://datascience.stackexchange.com/questions/75733/pca-for-complex-valued-data
def pca(X):
    mean = X.mean(axis=0) 
    center = X - mean 
    _, stds, pcs = np.linalg.svd(center/np.sqrt(X.shape[0])) 

    return stds**2, pcs

def get_dfs_(csi_data, 
             samp_rate = 1000,  # 采样率（Hz）
             window_size = 256,  # STFT的窗口长度
             nfft=1000,          # FFT 点数
             window_step = 10,   # 滑窗步长
             agg_type = 'pca',   # 子载波聚合策略：'pca' 或 'ms'
             n_pca=1,            # PCA主成分数（若使用pca聚合）
             log=False,           # 是否对谱图进行对数处理
             cache_folder=None):
    """
    :param csi_data: # 原始 CSI 数据，形如 [T, N]，T为时间长度，N为子载波或天线对
    ms: channel gain of each subcarrier
    """
    # start_time = time.time()   
    # if the file does not exist, compute it 
    # with record_function("compute_DFS"):
    #构造完整频率轴（包含负频率），并只保留 [-60Hz, 60Hz] 区间的频率分量作为低通滤波窗口。
    half_rate = samp_rate / 2
    uppe_stop = 60
    freq_bins_unwrap = np.concatenate((np.arange(0, half_rate, 1) / samp_rate, np.arange(-half_rate, 0, 1) / samp_rate))
    freq_lpf_sele = np.logical_and(np.less_equal(freq_bins_unwrap,(uppe_stop / samp_rate)),np.greater_equal(freq_bins_unwrap,(-uppe_stop / samp_rate)))
    freq_lpf_positive_max = 60

    if agg_type == 'pca' and csi_data.shape[1] >= 1:
        pca = PCA(n_components=n_pca)
        pca_coef = pca.fit_transform(np.absolute(np.transpose(csi_data, [1,0])))
        # [T,1]
        csi_data_agg = np.dot(csi_data, pca_coef)
        # always report the last pca component
        csi_data_agg = csi_data_agg[:,-1]
    elif agg_type == 'ms':
        # L1-normalize ms
        csi_data_agg = csi_data
    
    # DC removal
    csi_data_agg = csi_data_agg - np.mean(csi_data_agg, axis=0)
    noverlap = window_size - window_step
    freq, ticks, freq_time_prof_allfreq = signal.stft(csi_data_agg, fs=samp_rate, nfft=samp_rate,
                            window=('gaussian', window_size), nperseg=window_size, noverlap=noverlap, 
                            return_onesided=False,
                            padded=True)
    
    freq_time_prof_allfreq = np.array(freq_time_prof_allfreq)
    freq_time_prof = freq_time_prof_allfreq[freq_lpf_sele, :]

    if log:
        doppler_spectrum = np.log10(np.square(np.abs(freq_time_prof)) + 1e-20) + 20
        doppler_spectrum_phase = np.log10(np.square(np.angle(freq_time_prof)) + 1e-20) + 20
    else:
        # DO NOT USE widar3 version, will introduce interference in the frequency axis. making empty timeslots too large
        # doppler_spectrum = np.divide(abs(freq_time_prof), np.sum(abs(freq_time_prof), axis=0), out=np.zeros(freq_time_prof.shape), where=abs(freq_time_prof) != 0)
        # cal signal’s energy
        doppler_spectrum = np.square(np.abs(freq_time_prof))
        doppler_spectrum_phase =np.angle(freq_time_prof)
    # doppler_spectrum = np.divide(abs(doppler_spectrum), np.sum(abs(doppler_spectrum), axis=0), out=np.zeros(doppler_spectrum.shape), where=abs(doppler_spectrum) != 0)
    # doppler_spectrum = freq_time_prof
    # freq_bin = 0:freq_lpf_positive_max - 1 * freq_lpf_negative_min:-1]
    freq_bin = np.array(freq)[freq_lpf_sele]
    
    # shift the doppler spectrum to the center of the frequency bins
    # freq_time_prof_allfreq = [0, 1, 2 ... -2, -1]
    doppler_spectrum = np.roll(doppler_spectrum, freq_lpf_positive_max, axis=0)
    doppler_spectrum_phase = np.roll(doppler_spectrum_phase, freq_lpf_positive_max, axis=0)
    freq_bin = np.roll(freq_bin, freq_lpf_positive_max)
    return freq_bin, ticks, doppler_spectrum, doppler_spectrum_phase

def get_dfs_torch(csi_data, 
             samp_rate = 1000,  # 采样率（Hz）
             window_size = 256,  # STFT的窗口长度
             nfft=1000,          # FFT 点数
             window_step = 10,   # 滑窗步长
             agg_type = 'pca',   # 子载波聚合策略：'pca' 或 'ms'
             n_pca=1,            # PCA主成分数（若使用pca聚合）
             log=False,           # 是否对谱图进行对数处理
             cache_folder=None):
    """
    :param csi_data: # 原始 CSI 数据，形如 [T, N]，T为时间长度，N为子载波或天线对
    ms: channel gain of each subcarrier
    """
    # start_time = time.time()   
    # if the file does not exist, compute it 
    # with record_function("compute_DFS"):
    #构造完整频率轴（包含负频率），并只保留 [-60Hz, 60Hz] 区间的频率分量作为低通滤波窗口。
    half_rate = samp_rate / 2
    uppe_stop = 80
    freq_bins_unwrap = np.concatenate((np.arange(0, half_rate, 1) / samp_rate, np.arange(-half_rate, 0, 1) / samp_rate))
    freq_lpf_sele = np.logical_and(np.less_equal(freq_bins_unwrap,(uppe_stop / samp_rate)),np.greater_equal(freq_bins_unwrap,(-uppe_stop / samp_rate)))
    freq_lpf_positive_max = 80

    if agg_type == 'pca' and csi_data.shape[1] >= 1:
        pca = PCA(n_components=n_pca)
        pca_coef = pca.fit_transform(np.absolute(np.transpose(csi_data, [1,0])))
        # [T,1]
        csi_data_agg = np.dot(csi_data, pca_coef)
        # always report the last pca component
        csi_data_agg = csi_data_agg[:,-1]
    elif agg_type == 'ms':
        # L1-normalize ms
        csi_data_agg = csi_data
    
    # DC removal
    csi_data_agg = csi_data_agg - np.mean(csi_data_agg, axis=0)
    csi_data_tensor = torch.tensor(csi_data_agg, dtype=torch.cfloat)# 如果不是复数浮点型就会变成对称的

    std = window_size / 6  # 标准差的选择可根据需要调整
    gaussian_window = gaussian(M=window_size, std=std)
    gaussian_window_torch = torch.tensor(gaussian_window, dtype=torch.cfloat)
    n_fft = samp_rate  # 与原始 nfft 相同
    hop_length = window_step
    win_length = window_size
    noverlap = window_size - window_step
    Zxx = torch.stft(
            input=csi_data_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.signal.windows.gaussian(window_size, std=std),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=False,
            return_complex=True
        )
    # 频率轴
    freq = np.fft.fftfreq(n_fft, d=1/samp_rate)
    # 时间轴
    num_frames = Zxx.shape[1]
    ticks = np.arange(num_frames) * hop_length / samp_rate
    # 将复数张量转换为 NumPy 数组
    #Zxx_np = Zxx.detach().cpu().numpy()
    #freq_time_prof_allfreq = np.array(Zxx_np)
    freq_time_prof_allfreq = Zxx.detach().cpu().numpy()

    freq_time_prof = freq_time_prof_allfreq[freq_lpf_sele, :]


    if log:
        doppler_spectrum = np.log10(np.square(np.abs(freq_time_prof)) + 1e-20) + 20
        doppler_spectrum_phase = np.log10(np.square(np.angle(freq_time_prof)) + 1e-20) + 20
    else:
        # DO NOT USE widar3 version, will introduce interference in the frequency axis. making empty timeslots too large
        # doppler_spectrum = np.divide(abs(freq_time_prof), np.sum(abs(freq_time_prof), axis=0), out=np.zeros(freq_time_prof.shape), where=abs(freq_time_prof) != 0)
        # cal signal’s energy
        doppler_spectrum = np.square(np.abs(freq_time_prof))
        doppler_spectrum_phase =np.angle(freq_time_prof)
    # doppler_spectrum = np.divide(abs(doppler_spectrum), np.sum(abs(doppler_spectrum), axis=0), out=np.zeros(doppler_spectrum.shape), where=abs(doppler_spectrum) != 0)
    # doppler_spectrum = freq_time_prof
    # freq_bin = 0:freq_lpf_positive_max - 1 * freq_lpf_negative_min:-1]
    freq_bin = np.array(freq)[freq_lpf_sele]
    
    # shift the doppler spectrum to the center of the frequency bins
    # freq_time_prof_allfreq = [0, 1, 2 ... -2, -1]

    doppler_spectrum = np.roll(doppler_spectrum, freq_lpf_positive_max, axis=0)
    doppler_spectrum_phase = np.roll(doppler_spectrum_phase, freq_lpf_positive_max, axis=0)
    freq_bin = np.roll(freq_bin, freq_lpf_positive_max)
    return freq_bin, ticks, doppler_spectrum, doppler_spectrum_phase

def show_dfs(freq_bin, ticks, Zxx, file_path="doppler_spectrum_amp.png"):
    # 创建图像和坐标轴
    fig, ax = plt.subplots(figsize=(8, 4))

    # 绘制谱图（频率在纵轴，时间在横轴）
    mesh = ax.pcolormesh(
        ticks,                         # x轴：时间
        freq_bin,               # y轴：频率（Hz），单位转换
        Zxx,             # Z轴：强度
        shading='gouraud',             # 平滑填充
        cmap='jet'                     # 色图
    )

    # 添加颜色条
    fig.colorbar(mesh, ax=ax, label='Power')

    # 设置轴标签与标题
    ax.set_title('Doppler Spectrum')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # 显示图像
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)    

# 先改读数据 
# data_dir = "/data0/zzy25/widar3/r6_noprocess_complex/date_20181211_room_3_user_9_ges_Sweep_loc_5_ori_5_rx_r6_re_5.npy"
file_path_CSIDA="/opt/data/common/default/wifidata/csida/CSI_301/"
amp,pha,label=read_CSIDA_file_path(filepath=file_path_CSIDA)
print(f"amp shape is {amp.shape}")
print(f"pha shape is {pha.shape}")

# print(guess_phase_unit(pha))
csi_data=phase_deg_to_complex(amp,pha)
# csi_data=amp
# amp=amp[0,0,:,:]
# print(amp.shape)
# exit(0)
csi_data=csi_data[40,0,:,:]
print(csi_data.shape)


csi_data = csi_data.squeeze().reshape((1800,114))
csi_data_pha = np.angle(csi_data)
csi_data_amp = np.abs(csi_data)
#freq_bin, ticks, doppler_spectrum,doppler_spectrum_phase = get_dfs_(csi_data)
freq_bin, ticks, doppler_spectrum,doppler_spectrum_phase = get_dfs_torch(csi_data)

show_dfs(freq_bin, ticks, doppler_spectrum, file_path="doppler_spectrum_amp_torch2.png")
show_dfs(freq_bin, ticks, doppler_spectrum_phase, file_path="doppler_spectrum_phase_torch2.png")
