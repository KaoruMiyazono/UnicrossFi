import os
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

def process_and_split_data(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    all_data = []
    all_labels = []

    class_to_idx = {}
    idx_counter = 0
    cnt=0
    # 遍历所有文件，读取数据和类别标签
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            path = os.path.join(data_dir, filename)
            parts = filename.split('_')
            class_name = parts[3]  # 根据你说的类别位置
            if class_name not in class_to_idx:
                class_to_idx[class_name] = idx_counter
                idx_counter += 1
            print(f"Processing {filename} for class {class_name} cnt {cnt}")
            cnt=cnt+1
            data = torch.tensor(torch.load(path) if path.endswith('.pt') else torch.from_numpy(np.load(path)), dtype=torch.float)
            all_data.append(data)
            all_labels.append(class_to_idx[class_name])

    # 转换成 tensor
    all_data = torch.stack(all_data)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    # 先拆出 test，剩余的再拆 train/val
    data_train_val, data_test, labels_train_val, labels_test = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    data_train, data_val, labels_train, labels_val = train_test_split(
        data_train_val, labels_train_val, test_size=0.125, random_state=42, stratify=labels_train_val
    )  # 0.125 = 0.1 / 0.8 => 保证 val 是整体的10%

    # 保存 TensorDataset
    torch.save(TensorDataset(data_train, labels_train), os.path.join(save_dir, 'train.pt'))
    torch.save(TensorDataset(data_val, labels_val), os.path.join(save_dir, 'val.pt'))
    torch.save(TensorDataset(data_test, labels_test), os.path.join(save_dir, 'test.pt'))

    # 保存类别映射
    with open(os.path.join(save_dir, 'class_to_idx.txt'), 'w') as f:
        for k, v in class_to_idx.items():
            f.write(f"{k} {v}\n")

    print(f"Data saved to {save_dir}")
    print(f"Train size: {len(data_train)}, Val size: {len(data_val)}, Test size: {len(data_test)}")

# 使用示例
import numpy as np
data_dir = '/opt/data/private/ablation_study/data_widar_tdcsi_all/tdcsi_new'
save_dir = '/opt/data/private/ablation_study/data_widar_tdcsi_all'
process_and_split_data(data_dir, save_dir)
