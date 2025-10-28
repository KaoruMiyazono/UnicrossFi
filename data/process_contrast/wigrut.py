# TODO 把数据保存一下  
import numpy as np
import pickle
import os  
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
import scipy.io as sio

cnt=0 
# 注 保存格式 room_?_loc_?_user_?_cnt_?.png

def read_CSIDA(amp_file_path,pha_filepath,label_file_path):
    # pass
    # print(os.path.basename(amp_file_path))
    base_name=os.path.basename(amp_file_path)
    split_name=base_name.split("_")
    # print(split_name)

        
    with open(label_file_path,'rb') as f:
        label_data = pickle.load(f)
        # print(label_data.shape)
        
    with open(pha_filepath,'rb') as f:
        pha_data = pickle.load(f)
        # print(label_data.shape)
        # print(set(label_data))
    process_data=process_wigrut(pha_data,label_data,pha_filepath)
    print(pha_data.shape)
def save_single_rgb_image(img, save_path):
    """
    保存单张 RGB 图像到指定路径。
    
    参数：
    - img: 形状为 (h, w, 3) 的 numpy 数组，RGB 图像，值范围应为 [0, 1] 或 [0, 255]
    - save_path: 图像保存完整路径，包括文件名和后缀
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 如果图片像素是浮点类型且范围在0-1，转换成0-255 uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    img_pil = Image.fromarray(img)
    img_pil.save(save_path)


def read_CSIDA_file_path(filepath):
    """"读取CSIDA数据集数据地址"""
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if not file.startswith(".") and file.endswith("amp.pkl"):
                amp_filepath = os.path.join(root, file)
                label_filepath=amp_filepath.replace("amp.pkl","label.pkl") 
                pha_filepath=amp_filepath.replace("amp.pkl","pha.pkl")
                
                 
                print(amp_filepath)
                print(label_filepath)
                read_CSIDA(amp_filepath,pha_filepath,label_filepath)
                
                
def csi_to_rgb(data, cmap_name='jet'):
    """
    将形状为 (n, h, w) 的CSI数据转换为RGB图像
    cmap_name: 可选 'plasma', 'viridis', 'coolwarm', 'jet' 等
    """
    if data.ndim != 3:
        raise ValueError("输入 data 形状应为 (n, h, w)")
    
    n, h, w = data.shape
    rgb_images = []

    # 获取 colormap
    cmap = cm.get_cmap(cmap_name)

    for i in range(n):
        d = data[i]
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)  # 归一化到 [0, 1]
        rgb = cmap(d_norm)[:, :, :3]  # 返回的是 RGBA，取前3通道
        rgb_images.append(rgb)
    
    return np.stack(rgb_images)  # 返回 (n, h, w, 3)

   

def process_wigrut(data,label_data,pha_file_path):
    n,a,c,t=data.shape  
    # print(n,a,c,t)
    
    print(pha_file_path)
    data=data.reshape(n,a*c,t) #342 1800的2d矩阵 我需要把他转换为热图 而且是 黄紫配色 
    rgb=csi_to_rgb(data)
    print(rgb.shape)
    file_name=os.path.basename(pha_file_path)
    # print(file_name)
    file_name_split=file_name.split("_")
    print(file_name_split)
    room_id,loc_id,user_id=file_name_split[1],file_name_split[3],file_name_split[5]
    print(f"room_{room_id}_loc_{loc_id}_user_{user_id}")
    # print(room_id)
    global cnt 
    for i in range(rgb.shape[0]):
        this_data=rgb[i,:,:,:]
        this_label=label_data[i]
        # print(this_data.shape)
        # exit(0)
        file_path_this=f"ges_{this_label}_room_{room_id}_loc_{loc_id}_user_{user_id}_cnt_{i}"
        cnt=cnt+1
        save_single_rgb_image(this_data,save_path=f"/opt/data/private/ablation_study/data_wigrut/{file_path_this}.png")
        print(f"已经处理了{cnt}个样本")
    
    
    # exit(0)
    
    
def change_pkl_to_mat(amp_data,pha_data,label_data,save_path):
    """matlab无法读取pkl所以要转换成mat格式的文件 跑对比实验"""

    mat_data = {
        'amplitude': amp_data,
        'phase': pha_data,
        'label': label_data
    }

    # 保存为 .mat 文件
    sio.savemat(save_path, mat_data)
    print(f"已成功保存为 Matlab .mat 文件: {os.path.abspath(save_path)}")

def wigrunt_python():
    file_path_CSIDA="/opt/data/common/default/wifidata/csida/CSI_301/"

    for root, dirs, files in os.walk(file_path_CSIDA):
        for file in files:
            if not file.startswith(".") and file.endswith("amp.pkl"):
                amp_filepath = os.path.join(root, file)
                label_filepath=amp_filepath.replace("amp.pkl","label.pkl") 
                pha_filepath=amp_filepath.replace("amp.pkl","pha.pkl")

                with open(label_filepath,'rb') as f:
                    label_data = pickle.load(f)
                    # print(label_data.shape)
                    
                with open(pha_filepath,'rb') as f:
                    pha_data = pickle.load(f)

                with open(amp_filepath,'rb') as f:
                    amp_data = pickle.load(f)
                

                # print(file)
                file=file.replace("_amp.pkl",".mat")
                # print(file)
                root_dir_csida_wigrunt="/opt/data/private/ablation_study/data_mat_csida"
                print(f"amp_data.shape:{amp_data.shape} pha_data.shape {pha_data.shape} label_data.shape {label_data.shape}")
                # exit(0)
                change_pkl_to_mat(amp_data,pha_data,label_data,os.path.join(root_dir_csida_wigrunt,file))


wigrunt_python()

                    
                






    
    





# file_path_CSIDA="/data0/zzy25/CSIDA/CSIDA/CSI_301"  
# file_path_CSIDA="/opt/data/common/default/wifidata/csida/CSI_301/"

# read_CSIDA_file_path(file_path_CSIDA)