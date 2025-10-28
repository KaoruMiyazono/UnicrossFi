# TODO 找一个子载波选择算法
import numpy as np 
import os 
import pickle
import matplotlib.pyplot as plt
from PIL import Image

from pyts.image import RecurrencePlot
file_path_CSIDA_genearation="/opt/data/private/ablation_study/data_wisda_phase_CSIDA"
cnt=0
# /data0/zzy25/CSIDA/CSIDA/CSI_301 CSIDA
# 由于我要精确知道 每个域信息 所以 针对每张图 我的格式应该是 gesture_x_room_x_loc_x_user_x_at_x_sub_x.png这样子
def amp_image_to_heatmap(amp_image, filename='heatmap.png', cmap_name='coolwarm_r'):
    # 取出二维图像
    img = amp_image[0,:,:]

    # 归一化到0-255，防止数据范围问题
    img_norm = 255 * (img - img.min()) / (img.max() - img.min())
    img_uint8 = img_norm.astype(np.uint8)

    # 使用matplotlib的colormap转换成RGB，返回值是float32，范围0~1
    cmap = plt.get_cmap(cmap_name)
    img_color = cmap(img_uint8 / 255.0)[:, :, :3]  # 去掉alpha通道

    # 转成0-255的uint8
    img_color_uint8 = (img_color * 255).astype(np.uint8)

    # 用PIL保存为图片
    im_pil = Image.fromarray(img_color_uint8)
    im_pil.save(filename)
    # print(f'Heatmap saved as {filename}, size: {im_pil.size}')
def paa(ts, segments):
# paa时间序列 降维
    n = len(ts)
    if n == segments:
        return ts.copy()
    else:
        paa_result = np.zeros(segments)
        segment_size = n / segments
        for i in range(segments):
            start = int(np.ceil(i * segment_size))
            end = int(np.ceil((i + 1) * segment_size))
            paa_result[i] = ts[start:end].mean()
        return paa_result
    

def select_subcarry(amp_data):
    # pass 选择子载波
    antenna_num, subcarrier_num, time_num = amp_data.shape

    # 计算等间隔索引
    step = subcarrier_num // 9
    indices = np.arange(0, subcarrier_num, step)[:10]

    # 如果最后不够10个，重复最后一个索引
    if len(indices) < 10:
        indices = np.concatenate([indices, np.full(10 - len(indices), indices[-1])])
    print(f"选择的子载波索引: {indices}")
    # 选择
    selected = amp_data[:, indices, :]

    return selected,indices

def process_CSIDA(data,label,indices,file_name_dict):
    "处理CSI数据"
    print(data.shape)
    # for a in range(data.shape[0]):
    for s in range(data.shape[1]):
        a=1
        amp_a_s=data[a,s,:]
        # subcarry_name=indices[s]
        subcarry_name=s
        print(amp_a_s.shape)
        print(f"处理子载波: {subcarry_name}, 天线: {a}, 标签: {label}, 文件名信息: {file_name_dict}")
        paa_amp=paa(amp_a_s,224)
        rp=RecurrencePlot(dimension=1,time_delay=1)
        amp_image=rp.fit_transform(paa_amp.reshape(1,-1))
        print(amp_image.shape)
        filename=f"gesture_{label}_room_{file_name_dict['room']}_loc_{file_name_dict['loc']}_user_{file_name_dict['user']}_at_{a}_sub_{subcarry_name}_cnt_{file_name_dict['cnt_this']}.png"
        file_name_all=os.path.join(file_path_CSIDA_genearation,filename)
        amp_image_to_heatmap(amp_image,filename=file_name_all)


        # 读取确认尺寸
        img_read = plt.imread(file_name_all)
        print(file_name_all)
        print('Read image shape:', img_read.shape)  # 应该是 (224,224,4)
        global cnt
        cnt=cnt+1
        print(f"已处理 {cnt} 张图像")
            # exit(0)

    
    # exit(0)
    
def read_CSIDA(amp_file_path,label_file_path):
    # pass
    # print(os.path.basename(amp_file_path))
    base_name=os.path.basename(amp_file_path)
    split_name=base_name.split("_")
    print(split_name)
    with open(amp_file_path, 'rb') as f:
        amp_data = pickle.load(f)
        print(amp_data.shape)
    with open(label_file_path,'rb') as f:
        label_data = pickle.load(f)
        print(label_data.shape)
        # print(set(label_data))
    for i in range(amp_data.shape[0]): 
        # amp_this=
        amp_this=amp_data[i,:,:,:]
        # amp_this_selected,indices=select_subcarry(amp_this) #选择子载波
        indices=None
        amp_this_selected=amp_this
        lable_this=label_data[i]
        file_name_dict={'gesture':label_data[i],'room':split_name[1],'loc':split_name[3],'user':split_name[5],"cnt_this":i}
        process_CSIDA(amp_this_selected,lable_this,indices,file_name_dict)
        

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
                read_CSIDA(pha_filepath,label_filepath)
                
                # exit(0)

def cal_top5_sub_var(csi_ant_data):

    amp = np.abs(csi_ant_data)          # [C, T]
    var_per_sub = np.var(amp, axis=1)   # [C]
    
    # argsort 排序取最大的5个索引
    top5_indices = np.argsort(-var_per_sub)[:5]  # 注意负号实现降序
    
    return top5_indices

def read_widar_path(file_path_widar):
    cnt=0
    for file in os.listdir(file_path_widar):

        data=np.load(os.path.join(file_path_widar,file))
        # print(data.shape)
        # exit(0)
        amp_data=data[:,0,:,:]
        amp_data_att2=amp_data[1,:,:]

        top5= cal_top5_sub_var(amp_data_att2)
        

        for s in top5:
            # print(s)
            data_this_sub=amp_data_att2[s,:]
            paa_amp=paa(data_this_sub,224)
            print(paa_amp.shape)
            # exit(0)
            rp=RecurrencePlot(dimension=1,time_delay=1)
            amp_image=rp.fit_transform(paa_amp.reshape(1,-1))
            print(amp_image.shape)
            print(file)
            file_name_no_ext = file.replace(".npy", "")
            file_name_end=file_name_no_ext+f'_sub_{s}'+f".png"
            save_pre='/opt/data/private/ablation_study/data_tdcsi_widar_wisda_amp'
            filename=os.path.join(save_pre,file_name_end)
            if os.path.exists(filename):
                print(f"{filename}已经存在啊")
            else:
                amp_image_to_heatmap(amp_image,filename=os.path.join(save_pre,file_name_end))
            cnt=cnt+1
            print(f'已经处理了{cnt}个数据了！！！！')
            # print(data_this_sun.shape)
        # exit(0)
        
        
        
        
    print(cnt)
    
import os
import numpy as np
from scipy.io import savemat

def convert_npy_to_mat(npy_dir, mat_dir, key_name="data"):
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)

    for filename in os.listdir(npy_dir):
        if filename.endswith('.npy'):
            npy_path = os.path.join(npy_dir, filename)
            mat_filename = filename.replace('.npy', '.mat')
            mat_path = os.path.join(mat_dir, mat_filename)

            try:
                array = np.load(npy_path)
                savemat(mat_path, {key_name: array})
                print(f"Saved: {mat_path}, shape: {array.shape}")
                # exit(0)
            except Exception as e:
                print(f"Failed to process {npy_path}: {e}")




    
if __name__ == "__main__":
    # file_path_CSIDA="/opt/data/common/default/wifidata/csida/CSI_301/"  
    # read_CSIDA_file_path(file_path_CSIDA)

    file_path_widar='/opt/data/private/ablation_study/data_widar_800/tdcsi'
    
    read_widar_path(file_path_widar)

#     convert_npy_to_mat(
#     npy_dir="/opt/data/private/ablation_study/data_widar_fdaarc",
#     mat_dir="/opt/data/private/ablation_study/data_widar_mat"
# )