
import numpy as np 
import os 
from scipy.signal import resample
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # 必须导入以启用 3D 支持
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import pickle
from tqdm import tqdm


def read_CSIDA(root_dir,domain_prefix): 
    # 处理源域文件
    # 构建文件路径
    amp_path = os.path.join(root_dir, domain_prefix + "_CSIDA_amp.pkl")
    pha_path = os.path.join(root_dir, domain_prefix + "_CSIDA_pha.pkl")
    label_path = os.path.join(root_dir, domain_prefix + "_CSIDA_label.pkl")
    # 读取文件
    if os.path.exists(amp_path):
        with open(amp_path, 'rb') as f:
            amp = pickle.load(f)
    if os.path.exists(pha_path):
        with open(pha_path, 'rb') as f:
            phase = pickle.load(f)
    if os.path.exists(label_path):
        with open(label_path, 'rb') as f:
            label = pickle.load(f)
    # 检查是否成功加载
    if amp is None or phase is None:
        raise ValueError("Missing amp or phase data.")

    # 转为 numpy 格式（如果不是）
    amp = np.array(amp)
    phase = np.array(phase)
    # 重组为复数：CSI = amp * exp(j * phase)
    csi = amp * np.exp(1j * phase)  # shape 保持不变
    # 创建时间戳（使用最后一维作为时间轴）
    # 假设 shape 为 (N, C, T) 或 (..., T)
    time_axis = csi.shape[-1]
    timestamps = np.arange(time_axis)

    return csi, timestamps, label

class Intel:
    """
    class used to read csi data from .dat file
    This implementation is modified from
    https://github.com/citysu/csiread/blob/master/examples/csireadIntel5300.py
    """
    def __init__(self, file, nrxnum=3, ntxnum=2, pl_len=0, if_report=True):
        self.file = file
        self.nrxnum = nrxnum
        self.ntxnum = ntxnum
        self.pl_len = pl_len    # useless
        self.if_report = if_report #useless 
        # print(file)
        if not os.path.isfile(file):
            raise Exception("error: file does not exist, Stop!\n")

    def __getitem__(self, index):
        """Return contents of 0xbb packets"""
        ret = {
            "timestamp_low": self.timestamp_low[index],
            "bfee_count": self.bfee_count[index],
            "Nrx": self.Nrx[index],
            "Ntx": self.Ntx[index],
            "rssi_a": self.rssi_a[index],
            "rssi_b": self.rssi_b[index],
            "rssi_c": self.rssi_c[index],
            "noise": self.noise[index],
            "agc": self.agc[index],
            "perm": self.perm[index],
            "rate": self.rate[index],
            "csi": self.csi[index]
        }
        return ret

    def read(self):
        f = open(self.file, 'rb')
        if f is None:
            f.close()
            return -1

        lens = os.path.getsize(self.file)
        btype = np.int_
        #self.timestamp_low = np.zeros([lens//95], dtype = btype)
        self.timestamp_low = np.zeros([lens//95], dtype = np.int64)
        self.bfee_count = np.zeros([lens//95], dtype = btype)
        self.Nrx = np.zeros([lens//95], dtype = btype)
        self.Ntx = np.zeros([lens//95], dtype = btype)
        self.rssi_a = np.zeros([lens//95], dtype = btype)
        self.rssi_b = np.zeros([lens//95], dtype = btype)
        self.rssi_c = np.zeros([lens//95], dtype = btype)
        self.noise = np.zeros([lens//95], dtype = btype)
        self.agc = np.zeros([lens//95], dtype = btype)
        self.perm = np.zeros([lens//95, 3], dtype = btype)
        self.rate = np.zeros([lens//95], dtype = btype)
        self.csi = np.zeros([lens//95, 30, self.nrxnum, self.ntxnum], dtype = np.complex_) # type: ignore

        cur = 0
        count = 0
        while cur < (lens-3):
            temp = f.read(3)
            field_len = temp[1]+(temp[0]<<8)
            code = temp[2]
            cur += 3
            if code == 187:
                buf = f.read(field_len - 1)
                if len(buf) != field_len - 1:
                    break
                self.timestamp_low[count] = int.from_bytes(buf[:4], 'little')
                self.bfee_count[count] = int.from_bytes(buf[4:6], 'little')
                assert self.nrxnum == buf [8] # check the pre given nrx number is correct
                assert self.ntxnum == buf [9] # check the pre given ntx number is correct
                self.Nrx[count] = buf[8]
                self.Ntx[count] = buf[9]
                self.rssi_a[count] = buf[10]
                self.rssi_b[count] = buf[11]
                self.rssi_c[count] = buf[12]
                self.noise[count] = int.from_bytes(buf[13:14], 'little', signed=True)
                self.agc[count] = buf[14]
                self.rate[count] = int.from_bytes(buf[18:20], 'little')

                self.perm[count, 0] = buf[15] & 0x3
                self.perm[count, 1] = (buf[15] >> 2) & 0x3
                self.perm[count, 2] = (buf[15] >> 4) & 0x3

                index = 0
                payload = buf[20:]
                for i in range(30):
                    index += 3
                    remainder = index & 0x7
                    for j in range(buf[8]):
                        for k in range(buf[9]):
                            a = (payload[index // 8] >> remainder) | (payload[index // 8 + 1] << (8 - remainder)) & 0xff
                            b = (payload[index // 8 + 1] >> remainder) | (payload[index // 8 + 2] << (8 - remainder)) & 0xff
                            if a >= 128:
                                a -= 256
                            if b >= 128:
                                b -= 256
                            self.csi[count, i, self.perm[count, j], k] = a + b * 1.j
                            index += 16
                count += 1
            else:
                f.seek(field_len - 1, os.SEEK_CUR)
            cur += field_len - 1
        f.close()
        self.timestamp_low = self.timestamp_low[:count]
        self.bfee_count = self.bfee_count[:count]
        self.Nrx = self.Nrx[:count]
        self.Ntx = self.Ntx[:count]
        self.rssi_a = self.rssi_a[:count]
        self.rssi_b = self.rssi_b[:count]
        self.rssi_c = self.rssi_c[:count]
        self.noise = self.noise[:count]
        self.agc = self.agc[:count]
        self.perm = self.perm[:count, :]
        self.rate = self.rate[:count]
        self.csi = self.csi[:count, :, :, :]
        self.count = count

    def get_total_rss(self):
        """Calculates the Received Signal Strength (RSS) in dBm from CSI"""
        rssi_mag = np.zeros_like(self.rssi_a, dtype=float)
        rssi_mag += self.__dbinvs(self.rssi_a)
        rssi_mag += self.__dbinvs(self.rssi_b)
        rssi_mag += self.__dbinvs(self.rssi_c)
        ret = self.__db(rssi_mag) - 44 - self.agc
        return ret

    def get_scaled_csi(self):
        """Converts CSI to channel matrix H"""
        csi = self.csi
        csi_sq = (csi * csi.conj()).real
        csi_pwr = np.sum(csi_sq, axis=(1, 2, 3))
        rssi_pwr = self.__dbinv(self.get_total_rss())

        scale = rssi_pwr / (csi_pwr / 30)

        noise_db = self.noise
        thermal_noise_pwr = self.__dbinv(noise_db)
        thermal_noise_pwr[noise_db == -127] = self.__dbinv(-92)

        quant_error_pwr = scale * (self.Nrx * self.Ntx)
        total_noise_pwr = thermal_noise_pwr + quant_error_pwr

        ret = self.csi * np.sqrt(scale / total_noise_pwr).reshape(-1, 1, 1, 1)
        ret[self.Ntx == 2] *= np.sqrt(2)
        ret[self.Ntx == 3] *= np.sqrt(self.__dbinv(4.5))
        ret = ret.conj()
        return ret

    def get_scaled_csi_sm(self):
        """Converts CSI to channel matrix H
        This version undoes Intel's spatial mapping to return the pure
        MIMO channel matrix H.
        """
        ret = self.get_scaled_csi()
        ret = self.__remove_sm(ret)
        return ret

    def __dbinvs(self, x):
        """Convert from decibels specially"""
        ret = np.power(10, x / 10)
        ret[ret == 1] = 0
        return ret

    def __dbinv(self, x):
        """Convert from decibels"""
        ret = np.power(10, x / 10)
        return ret

    def __db(self, x):
        """Calculates decibels"""
        ret = 10 * np.log10(x)
        return ret

    def __remove_sm(self, scaled_csi):
        """Actually undo the input spatial mapping"""
        sm_1 = 1
        sm_2_20 = np.array([[1, 1],
                            [1, -1]]) / np.sqrt(2)
        sm_2_40 = np.array([[1, 1j],
                            [1j, 1]]) / np.sqrt(2)
        sm_3_20 = np.array([[-2 * np.pi / 16, -2 * np.pi / (80 / 33), 2 * np.pi / (80 / 3)],
                            [ 2 * np.pi / (80 / 23), 2 * np.pi / (48 / 13), 2 * np.pi / (240 / 13)],
                            [-2 * np.pi / (80 / 13), 2 * np.pi / (240 / 37), 2 * np.pi / (48 / 13)]])
        sm_3_20 = np.exp(1j * sm_3_20) / np.sqrt(3)
        sm_3_40 = np.array([[-2 * np.pi / 16, -2 * np.pi / (80 / 13), 2 * np.pi / (80 / 23)],
                            [-2 * np.pi / (80 / 37), -2 * np.pi / (48 / 11), -2 * np.pi / (240 / 107)],
                            [ 2 * np.pi / (80 / 7), -2 * np.pi / (240 / 83), -2 * np.pi / (48 / 11)]])
        sm_3_40 = np.exp(1j * sm_3_40) / np.sqrt(3)
    
        ret = scaled_csi
        for i in range(self.count):
            M = self.Ntx[i]
            if (int(self.rate[i]) & 2048) == 2048:
                if M == 3:
                    sm = sm_3_40
                elif M == 2:
                    sm = sm_2_40
                else:
                    sm = sm_1
            else:
                if M == 3:
                    sm = sm_3_20
                elif M == 2:
                    sm = sm_2_20
                else:
                    sm = sm_1
            if sm != 1:
                ret[i, :, :, :M] = ret[i, :, :, :M].dot(sm.T.conj())
        return ret



def TD_CSI_process_widar(file_path, root_dir): # TODO 
    # 先读取，然后天线选择，然后ratio，然后TDCSI，然后resize后保存NPY，然后统一归一化
    target_gestures = ["Push&Pull", "Sweep", "Clap", "Slide", "DrawO(Horizontal)", "DrawZigzag(Horizontal)"]
    with open(file_path, 'r') as f:
        lines = f.readlines()

    min_csi_len = float('inf')
    max_csi_len = 0
    min_td_len = float('inf')
    max_td_len = 0

    for line in tqdm(lines, desc="Processing WiDAR samples"):
        line = line.strip()
        if '->' not in line:
            continue
        src, dst = [x.strip() for x in line.split('->')]
        # 示例：20181128-user6-Clap-3-3-2-r2-room2.dat
        gesture = dst.split('-')[2]
        room=dst.split('-')[7]
        room=room.removesuffix(".dat")
        if gesture in target_gestures:
            print(f"Accepted: {line}")
            # 读取数据
            widar_data_single = Intel(file=src, nrxnum=3, ntxnum=1, pl_len=0, if_report=True)
            widar_data_single.read()
            csi = widar_data_single.get_scaled_csi() # shape [T,C,A,1]
            time_stamp = widar_data_single.timestamp_low # shape [T]
            csi = np.squeeze(csi)  # 去掉最后一个维度 -> shape: [T, C, A]
            if csi.shape[0] == 0: # 当前文件时长为0，直接跳过
                continue
            csi = np.transpose(csi, (2, 1, 0))  # 调整维度顺序: [A, C, T]
            ## 天线选择
            stability_score = cal_ant_select_coeff(csi)
            # Select antenna 
            all_ants = {0, 1, 2}
            sel_ant1 = int(np.argmax(stability_score))
            ref_ant2 = int(np.argmin(stability_score))
            mid_ant3 = (all_ants - {sel_ant1, ref_ant2}).pop()
            csi_sel1 = csi[sel_ant1,:,:].squeeze()
            csi_ref2 = csi[ref_ant2,:,:].squeeze()
            csi_mid3 = csi[mid_ant3,:,:].squeeze()
            # csi ratio
            csi_ratio1 = csi_sel1 / (csi_ref2 + 1e-8)       # [C, T]
            csi_ratio3 = csi_mid3 / (csi_ref2 + 1e-8)       # [C, T]
            # 统计csiratio后的时间长度
            current_len = csi_ratio1.shape[-1]
            min_csi_len = min(min_csi_len, current_len)
            max_csi_len = max(max_csi_len, current_len)
            # TD-CSI
            tdcsi1 = TD_computing(csi_ratio1,time_stamp,interval=80000) # interval单位miu second # shape  [C, T]
            tdcsi3 = TD_computing(csi_ratio3,time_stamp,interval=80000)
            # 如果 tdcsi1 是 list 类型，跳过
            if isinstance(tdcsi1, list) or isinstance(tdcsi3, list):
                print(f"[WARNING] Empty TD-CSI, skipping {src}")
                continue
            # 统计td后的时间长度
            current_len = tdcsi1.shape[-1]
            min_td_len = min(min_td_len, current_len)
            max_td_len = max(max_td_len, current_len)
            # 针对复数的resize
            tdcsi1 = resample_complex_signal_without_timestamp(tdcsi1,800)  # [C, T]
            tdcsi3 = resample_complex_signal_without_timestamp(tdcsi3,800)
            # 拼接回去
            tdcsi = np.stack([tdcsi1, tdcsi3], axis=0)  # [2, C, T]
            # 计算振幅和相位后将其combine到一起
            amp_tdcsi = np.abs(tdcsi)   # [2, C, T]
            pha_tdcsi = np.angle(tdcsi)
            combine_tdcsi = np.stack([amp_tdcsi, pha_tdcsi], axis=1)

            # 保存npy数据
            # 在这之后进行统一的归一化
            file_name = os.path.basename(src)  # 提取文件名
            date = dst.split('-')[0]  # 提取日期部分
            parts = file_name.split('-')
            # 提取各个部分
            user_id = parts[0]                # User ID (e.g., 'user6')
            gesture_type = gesture             # Gesture type (e.g., 'Clap')
            location = parts[2]                # Location (e.g., '3')
            orientation = parts[3]             # Orientation (e.g., '3')
            repetition_number = parts[4]       # Repetition (e.g., '2')
            save_name = f"{date}_room_{room}_{gesture_type}_user_{user_id}_location_{location}_orientation_{orientation}_repetition_{repetition_number}.npy"
            # 创建两个子目录
            tdcsi_dir = os.path.join(root_dir, "tdcsi_new")
            os.makedirs(tdcsi_dir, exist_ok=True)
            # 保存 conj 数据
            tdcsi_save_path = os.path.join(tdcsi_dir, save_name)
            np.save(tdcsi_save_path, combine_tdcsi)
            print(f"Saved {save_name}")
        else:
            print(f"Filtered out: {gesture}")

    print(f"Minimum TD-CSI length: {min_csi_len}")
    print(f"Maximum TD-CSI length: {max_csi_len}")
    print(f"Minimum TD-CSI length: {min_td_len}")
    print(f"Maximum TD-CSI length: {max_td_len}")
    

def TD_CSI_process_widar_all(file_path, root_dir): # TODO 
    # 先读取，然后天线选择，然后ratio，然后TDCSI，然后resize后保存NPY，然后统一归一化
    target_gestures = ["Push&Pull", "Sweep", "Clap", "Slide", "DrawO(Horizontal)", "DrawZigzag(Horizontal)"]
    with open(file_path, 'r') as f:
        lines = f.readlines()

    min_csi_len = float('inf')
    max_csi_len = 0
    min_td_len = float('inf')
    max_td_len = 0

    for line in tqdm(lines, desc="Processing WiDAR samples"):
        line = line.strip()
        if '->' not in line:
            continue
        src, dst = [x.strip() for x in line.split('->')]
        # 示例：20181128-user6-Clap-3-3-2-r2-room2.dat
        gesture = dst.split('-')[2]
        room=dst.split('-')[7]
        room=room.removesuffix(".dat")
        # if gesture in target_gestures:
        print(f"Accepted: {line}")
        # 读取数据
        widar_data_single = Intel(file=src, nrxnum=3, ntxnum=1, pl_len=0, if_report=True)
        widar_data_single.read()
        csi = widar_data_single.get_scaled_csi() # shape [T,C,A,1]
        time_stamp = widar_data_single.timestamp_low # shape [T]
        csi = np.squeeze(csi)  # 去掉最后一个维度 -> shape: [T, C, A]
        if csi.shape[0] == 0: # 当前文件时长为0，直接跳过
            continue
        csi = np.transpose(csi, (2, 1, 0))  # 调整维度顺序: [A, C, T]
        ## 天线选择
        stability_score = cal_ant_select_coeff(csi)
        # Select antenna 
        all_ants = {0, 1, 2}
        sel_ant1 = int(np.argmax(stability_score))
        ref_ant2 = int(np.argmin(stability_score))
        mid_ant3 = (all_ants - {sel_ant1, ref_ant2}).pop()
        csi_sel1 = csi[sel_ant1,:,:].squeeze()
        csi_ref2 = csi[ref_ant2,:,:].squeeze()
        csi_mid3 = csi[mid_ant3,:,:].squeeze()
        # csi ratio
        csi_ratio1 = csi_sel1 / (csi_ref2 + 1e-8)       # [C, T]
        csi_ratio3 = csi_mid3 / (csi_ref2 + 1e-8)       # [C, T]
        # 统计csiratio后的时间长度
        current_len = csi_ratio1.shape[-1]
        min_csi_len = min(min_csi_len, current_len)
        max_csi_len = max(max_csi_len, current_len)
        # TD-CSI
        tdcsi1 = TD_computing(csi_ratio1,time_stamp,interval=80000) # interval单位miu second # shape  [C, T]
        tdcsi3 = TD_computing(csi_ratio3,time_stamp,interval=80000)
        # 如果 tdcsi1 是 list 类型，跳过
        if isinstance(tdcsi1, list) or isinstance(tdcsi3, list):
            print(f"[WARNING] Empty TD-CSI, skipping {src}")
            continue
        # 统计td后的时间长度
        current_len = tdcsi1.shape[-1]
        min_td_len = min(min_td_len, current_len)
        max_td_len = max(max_td_len, current_len)
        # 针对复数的resize
        tdcsi1 = resample_complex_signal_without_timestamp(tdcsi1,800)  # [C, T]
        tdcsi3 = resample_complex_signal_without_timestamp(tdcsi3,800)
        # 拼接回去
        tdcsi = np.stack([tdcsi1, tdcsi3], axis=0)  # [2, C, T]
        # 计算振幅和相位后将其combine到一起
        amp_tdcsi = np.abs(tdcsi)   # [2, C, T]
        pha_tdcsi = np.angle(tdcsi)
        combine_tdcsi = np.stack([amp_tdcsi, pha_tdcsi], axis=1)

        # 保存npy数据
        # 在这之后进行统一的归一化
        file_name = os.path.basename(src)  # 提取文件名
        date = dst.split('-')[0]  # 提取日期部分
        parts = file_name.split('-')
        # 提取各个部分
        user_id = parts[0]                # User ID (e.g., 'user6')
        gesture_type = gesture             # Gesture type (e.g., 'Clap')
        location = parts[2]                # Location (e.g., '3')
        orientation = parts[3]             # Orientation (e.g., '3')
        repetition_number = parts[4]       # Repetition (e.g., '2')
        save_name = f"{date}_room_{room}_{gesture_type}_user_{user_id}_location_{location}_orientation_{orientation}_repetition_{repetition_number}.npy"
        # 创建两个子目录
        tdcsi_dir = os.path.join(root_dir, "tdcsi_new")
        os.makedirs(tdcsi_dir, exist_ok=True)
        # 保存 conj 数据
        tdcsi_save_path = os.path.join(tdcsi_dir, save_name)
        np.save(tdcsi_save_path, combine_tdcsi)
        print(f"Saved {save_name}")
    # else:
    #     print(f"Filtered out: {gesture}")

    print(f"Minimum TD-CSI length: {min_csi_len}")
    print(f"Maximum TD-CSI length: {max_csi_len}")
    print(f"Minimum TD-CSI length: {min_td_len}")
    print(f"Maximum TD-CSI length: {max_td_len}")


def TD_computing(csiratio, timestamp, interval=80000):
    """
    根据给定 interval 和非均匀 timestamp 执行间隔跳跃式差分。
    Args:
        csiratio (np.ndarray): CSI 比值数据，shape = [C, T]
        timestamp (np.ndarray): 时间戳数组，单位微秒，shape = [T]
        interval (float): 目标间隔（秒）
    Returns:
        np.ndarray: 差分后结果，shape = [A, C, M]，M ≤ T，表示有效差分数
    """
    C, T = csiratio.shape
    assert timestamp.shape[0] == T, "timestamp 和 csiratio 时间维度不一致"
    diff_list = []
    for i in range(T):
        t0 = timestamp[i]
        # 找第一个满足条件的后续点
        j = np.searchsorted(timestamp, t0 + interval, side='left') # 如果没找到会返回len(timestamp),所以有下面的判定语句
        if j < T: 
            diff = csiratio[:, j] - csiratio[:, i]  # shape: [C]
            diff_list.append(diff[..., np.newaxis])       # 添加一个新维度作为时间维
    if not diff_list:
        print('# 没有任何有效差分')
        return diff_list
    # 拼接为 [C, M]
    tdcsi = np.concatenate(diff_list, axis=-1)
    return tdcsi

def resample_complex_signal_without_timestamp(h, T_new):
    """
    对复数信号 h 进行时间维度重采样
    """
    # TODO 对低于800的直接padding

    C, T = h.shape
    resampled_h = np.zeros((C, T_new), dtype=np.complex128)
    for c in range(C):
        resampled_h[c] = resample(h[c], T_new)  # scipy 会自动处理复数
    return resampled_h

def cal_ant_select_coeff(csi_data):
    """
        csi_data (np.ndarray): CSI data of shape [A, C, T] 
                               (Antennas, Subcarriers, Timesteps)
    """
    amp = np.abs(csi_data)  # [A, C, T]
    # Mean and variance over time axis
    mean_amp = np.mean(amp, axis=2)  # [A, C]
    var_amp = np.var(amp, axis=2)    # [A, C]
    # Stability score: mean / var (add epsilon to avoid division by zero)
    stability = mean_amp / (var_amp + 1e-8)  # [A, C]
    # Average over subcarriers
    stability_score = np.mean(stability, axis=1)  # [A]

    return stability_score

# import numpy as np


def cal_sub_var(csi_ant_data):
    """
        csi_ant_data (np.ndarray): CSI data of shape [C, T] 
                                   (Subcarriers, Timesteps) for one antenna.
    """
    amp = np.abs(csi_ant_data)        # shape [C, T]
    var_per_sub = np.var(amp, axis=1) # shape [C], variance over time
    selected_sub = int(np.argmax(var_per_sub))
    return selected_sub

## 对数据进行归一化以后重新保存
# 保证amp和phase在同一个尺度上进行处理
def norm_all_csidata(folder_path):
    filelist = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    print(f"Processing {len(filelist)} files in {folder_path}")

    for fname in tqdm(filelist, desc="Normalizing CSI data"):
        fpath = os.path.join(folder_path, fname)
        data = np.load(fpath)  # [2, 2, C, T]
        
        # 安全检查
        if data.shape[1] != 2:
            print(f"Skipping invalid file: {fname}")
            continue

        amp = data[:, 0, :, :]  # [2, C, T]
        pha = data[:, 1, :, :]  # [2, C, T]  -pi 到 pi
        # 幅值归一化到 [0,pi]
        amp_min = np.min(amp)
        amp_max = np.max(amp)
        amp_norm = (amp - amp_min) / (amp_max - amp_min + 1e-8)
        amp_norm = np.pi*amp_norm
        # 合并
        data_norm = np.stack([amp_norm, pha], axis=1)  # [ant=2, 2, C, T]
        # 保存覆盖原文件
        np.save(fpath, data_norm)
    print("[✓] All CSI data normalized and saved.")

def to_float32(folder_path):
    filelist = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    print(f"Processing {len(filelist)} files in {folder_path}")
    for fname in tqdm(filelist, desc="Normalizing CSI data"):
        fpath = os.path.join(folder_path, fname)
        data = np.load(fpath).astype(np.float32)
        # 保存覆盖原文件
        np.save(fpath, data)

if __name__ == "__main__":
    file_path = "/opt/data/private/FDAARC/data/widar_stat/updated_output.txt"
    # root_dir="/opt/data/private/ablation_study/data_widar_800"
    root_dir="/opt/data/private/ablation_study/data_widar_tdcsi_all"
    

    # TD_CSI_process_widar(file_path, root_dir)
    # TD_CSI_process_widar_all(file_path, root_dir)

    # norm_all_csidata("/opt/data/private/ablation_study/data_widar_800/tdcsi_new")
    norm_all_csidata("/opt/data/private/ablation_study/data_widar_tdcsi_all/tdcsi_new")
    
    # to_float32("/opt/data/private/ablation_study/data_widar_800/tdcsi_new")
    to_float32("/opt/data/private/ablation_study/data_widar_tdcsi_all/tdcsi_new")
    


