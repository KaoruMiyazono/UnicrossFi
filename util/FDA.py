
import torch

def from_ampphase_to_complex(x):
    amp = x[..., 0]                     # 幅度
    phase = x[..., 1]                   # 相位（单位为弧度）
    real = amp * torch.cos(phase)      # 实部
    imag = amp * torch.sin(phase)      # 虚部
    x_complex = torch.complex(real, imag)  # 得到复数张量，shape 为 [..., T]
    return x_complex

def to_ampphase(x: torch.Tensor) -> torch.Tensor:
    """
    将复数张量 [B, A, S, T] 转换为 [B, A, S, T, 2]，最后一维为 [amp, phase]
    
    参数:
        x: torch.complex64 或 complex128 类型的张量，形状为 [B, A, S, T]
        
    返回:
        torch.float32 张量，形状为 [B, A, S, T, 2]
    """
    amp = torch.abs(x)                      # [B, A, S, T]
    phase = torch.angle(x)                  # [B, A, S, T]
    return torch.stack([amp, phase], dim=-1)  # → [B, A, S, T, 2]

def bandwidth_freq_mutate_1d_with_fs(src, trg, fs=1000, cutoff_freq_lower=10,cutoff_freq_upper=100):
    """ 
    Args:
        src: 源幅度谱 (B, C, L)
        trg: 目标幅度谱 
        fs: 采样频率 (Hz)
        cutoff_freq: 截止频率 (Hz)
    """
    t = src.shape[-1]
    nyquist = fs / 2  # Nyquist频率
    total_freq_bins = t  # FFT后的频点总数（rfft的特殊性需处理）
    # print(B,C,t)
    # 计算截止频率对应的频点位置
    cutoff_bin_lower = int( (cutoff_freq_lower / nyquist) * t )  # 只考虑单侧频谱
    cutoff_bin_upper = int( (cutoff_freq_upper / nyquist) * t )  # 只考虑单侧频谱
    # print(cutoff_bin)
    
    # 替换高频区域（考虑rfft的对称性）
    src[..., cutoff_bin_lower:cutoff_bin_upper] = trg[..., cutoff_bin_lower:cutoff_bin_upper]
    return src
def high_freq_mutate_1d_with_fs(amp_src, amp_trg, fs=1000, cutoff_freq=300):
    """ 使用物理频率控制低频区域
    Args:
        amp_src: 源幅度谱 (B, C, L)
        amp_trg: 目标幅度谱 
        fs: 采样频率 (Hz)
        cutoff_freq: 截止频率 (Hz)
    """
    t = amp_src.shape[-1]
    nyquist = fs / 2  # Nyquist频率
    total_freq_bins = t  # FFT后的频点总数（rfft的特殊性需处理）
    # print(B,C,t)
    # 计算截止频率对应的频点位置
    cutoff_bin = int( (cutoff_freq / nyquist) * t )  # 只考虑单侧频谱
    # print(cutoff_bin)
    
    # 替换高频区域（考虑rfft的对称性）
    amp_src[..., cutoff_bin:] = amp_trg[..., cutoff_bin:]
    return amp_src

def FDA_1d_with_fs(src_signal, trg_signal, fs=1000, cutoff_freq=300,cutoff_freq_upper=None):
    """
    Args:
        src_signal: 源信号 (B, C, T)
        trg_signal: 目标信号 (B, C, T)
        fs: 采样频率 (Hz)
        cutoff_freq: 截止频率 (Hz)
        cutoff_freq_upper: 截止频率上限 (Hz)
    """
    # 输入可以为任意形状: (B, ?, T),仅需要指明针对最后一维度进行FDA即可
    src = src_signal
    trg = trg_signal
    # 计算RFFT（实数信号FFT）
    fft_src = torch.fft.rfft(src, dim=-1, norm='forward')  # 输出形状 (B, C, L+1)
    fft_trg = torch.fft.rfft(trg, dim=-1, norm='forward')
    # rfreqs = torch.fft.rfftfreq(1800, 1/1000)  # 得到非负频率
    # 分解幅度和相位
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg,pha_trg = torch.abs(fft_trg),torch.angle(fft_trg)
    # 使用物理频率进行高频替换
    # amp_src_mutated = high_freq_mutate_1d_with_fs(amp_src, amp_trg, fs=fs, cutoff_freq=cutoff_freq)
    if cutoff_freq_upper==None:
        pha_src_mutated = high_freq_mutate_1d_with_fs(pha_src, pha_trg, fs=fs, cutoff_freq=cutoff_freq)
    else:
        pha_src_mutated=bandwidth_freq_mutate_1d_with_fs(pha_src, pha_trg,fs=fs,cutoff_freq_lower=cutoff_freq,cutoff_freq_upper=cutoff_freq_upper)
    
    # 重建信号
    fft_mixed = torch.polar(amp_src, pha_src_mutated)
    mixed = torch.fft.irfft(fft_mixed, n=src.size(-1), dim=-1, norm='forward')
    
    return mixed.to(src_signal.dtype)

def FDA_complex(src_signal, trg_signal, fs=1000, cutoff_freq=300,cutoff_freq_upper=None):
    # 输入可以为任意形状: (B, ?, T),仅需要指明针对最后一维度进行FDA即可
    src = src_signal # complex data
    trg = trg_signal

    # 计算RFFT（实数信号FFT）
    fft_src = torch.fft.fft(src, dim=-1, norm='forward')  # 输出形状 (B, ?, L)
    fft_trg = torch.fft.fft(trg, dim=-1, norm='forward')
    # 分解幅度和相位
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg,pha_trg = torch.abs(fft_trg),torch.angle(fft_trg)
    
    # 使用物理频率进行高频替换
    # amp_src_mutated = high_freq_mutate_1d_with_fs(amp_src, amp_trg, fs=fs, cutoff_freq=cutoff_freq)
    if cutoff_freq_upper==None:
        pha_src_mutated = high_freq_mutate_1d_with_fs(pha_src, pha_trg, fs=fs, cutoff_freq=cutoff_freq)
    else:
        pha_src_mutated=bandwidth_freq_mutate_1d_with_fs(pha_src, pha_trg,fs=fs,cutoff_freq_lower=cutoff_freq,cutoff_freq_upper=cutoff_freq_upper)
    
    # 重建信号
    fft_mixed = torch.polar(amp_src, pha_src_mutated)
    mixed = torch.fft.ifft(fft_mixed, n=src.size(-1), dim=-1, norm='forward')
    
    return mixed.to(src_signal.dtype)


def batch_pca_pytorch(data: torch.Tensor, n_pca: int):
    """
    对每个样本 (C, T) 沿 C 做 PCA，输出 (B, n_pca, T)
    :param data: Tensor of shape (B, C, T), float32 or float64, on GPU or CPU
    :param n_pca: number of principal components to keep
    :return: Tensor of shape (B, n_pca, T)
    """
    B, C, T = data.shape
    device = data.device
    dtype = data.dtype
    # 1. 去中心化（在时间维，因为这个PCA是将时间点看作样本点）
    mean = data.mean(dim=-1, keepdim=True)  # (B, 1, T)
    x_centered = data - mean               # (B, C, T)
    # 2. 计算协方差矩阵（每个样本）
    #    cov = X @ X^T / T
    cov = torch.matmul(x_centered, x_centered.transpose(1, 2)) / T  # shape: (B, C, C)
    # 3. SVD 分解协方差矩阵（只保留特征向量）
    #    U: (B, C, C)
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
    # 4. 取前 n_pca 个主成分方向
    U_reduced = U[:, :, :n_pca]  # (B, C, n_pca)
    # 5. 将原始数据投影到主成分方向上
    #    U^T @ X_centered → (B, n_pca, T)
    x_pca = torch.matmul(U_reduced.transpose(1, 2), x_centered)
    return x_pca  # shape: (B, n_pca, T)


def STFDA_1d_with_fs(src_signal, trg_signal, 
                     window_size=256, window_step=10,
                     samp_rate = 1000,cutoff_freq=300,cutoff_freq_upper=None,
                     pca_flag=False):
    """
    Args:
        src_signal: 源信号 (B, C, T)
        trg_signal: 目标信号 (B, C, T)
        window_size: 窗口大小
        window_step: 窗口步长
        samp_rate: 采样频率 (Hz)
        cutoff_freq: 截止频率 (Hz)
        cutoff_freq_upper: 截止频率上限 (Hz)
        pca_flag: 是否使用PCA聚合
    """
    # 输入形状: (B, C, T)
    src = src_signal
    trg = trg_signal
    if pca_flag:
        src = batch_pca_pytorch(src, 1)
        trg = batch_pca_pytorch(trg, 1)
    # 计算复数信号STFT
    Zxx_src = torch.stft(
            input=src,
            n_fft=samp_rate,
            hop_length=window_step,
            win_length=window_size,
            window=torch.signal.windows.gaussian(window_size, std=window_size / 6),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=False,
            return_complex=True
        )  # (B, n_fft, 滑动窗口数量)
    Zxx_trg = torch.stft(
            input=trg,
            n_fft=samp_rate,
            hop_length=window_step,
            win_length=window_size,
            window=torch.signal.windows.gaussian(window_size, std=window_size / 6),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=False,
            return_complex=True
        )
    # rfreqs = torch.fft.rfftfreq(1800, 1/1000)  # 得到非负频率
    # 分解幅度和相位
    amp_src, pha_src = torch.abs(Zxx_src), torch.angle(Zxx_src)
    amp_trg,pha_trg = torch.abs(Zxx_trg),torch.angle(Zxx_trg)
    
    # 使用物理频率进行高频替换
    # amp_src_mutated = high_freq_mutate_1d_with_fs(amp_src, amp_trg, fs=fs, cutoff_freq=cutoff_freq)
    if cutoff_freq_upper==None:
        pha_src_mutated = high_freq_mutate_1d_with_fs(pha_src, pha_trg, fs=samp_rate, cutoff_freq=cutoff_freq)
    else:
        pha_src_mutated=bandwidth_freq_mutate_1d_with_fs(pha_src, pha_trg,fs=samp_rate,cutoff_freq_lower=cutoff_freq,cutoff_freq_upper=cutoff_freq_upper)
    
    # 重建信号
    stft_mixed = torch.polar(amp_src, pha_src_mutated)
    mixed = torch.fft.irfft(stft_mixed, n=src.size(-1), dim=-1, norm='forward')
    
    return mixed.to(src_signal.dtype)

# B, C, T = 2, 342, 1800
# src = torch.randn(B, C, T)
# trg = torch.randn(B, C, T)

# fs = 1000  # 采样频率
# cutoff_freq = 7  # 截止频率(Hz)
# mixed_fs = FDA_1d_with_fs(src, trg, fs=fs, cutoff_freq=cutoff_freq)
# print(mixed_fs.shape)
# print(f"实际替换频率范围：0-{cutoff_freq}Hz (Nyquist={fs//2}Hz)")