import os
import numpy as np
import pandas as pd
import cv2
import sys
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 读取数据
IDS_TYPE = sys.argv[1]
ini_df = pd.read_csv("Ini/" + IDS_TYPE + "ini.csv")
pred1_df = pd.read_csv("Pred1/" + IDS_TYPE + "pred1.csv")
pred2_df = pd.read_csv("Pred2/" + IDS_TYPE + "pred2.csv")

# 确保保存路径存在
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

# 目标特征列
feature_columns = ['length', 'inter_arrival_time', 'rolling_mean_length', 'rolling_std_length', 'rolling_max_length', 'rolling_min_length']

# 用对数的方法来归一化
def normalize_and_scale(data):
    if data.size == 0 or np.isnan(data).any():
        return None  

    scaled_data = np.zeros_like(data, dtype=np.uint8)  # 预分配数组
    for col in range(data.shape[1]):  # 遍历每一列
        col_data = data[:, col]
        min_val, max_val = np.min(col_data), np.max(col_data)
        
        # 归一化，防止 min == max 导致的 NaN
        if max_val > min_val:
            scaled_data[:, col] = ((col_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            scaled_data[:, col] = 128  # 若数据恒定，填充为灰色（避免黑块）
    
    return scaled_data

# 处理每个 flow_id
flow_ids = ini_df['flow_id'].unique()  # 获取所有 flow_id
generated_images = 0  # 统计成功生成的图片数量

for flow_id in tqdm(flow_ids, desc="Processing flows", unit="flow"):
    # 获取当前 flow_id 的数据
    ini_group = ini_df[ini_df['flow_id'] == flow_id]
    ini_data = ini_group[feature_columns].values
    pred1_data = pred1_df[pred1_df['flow_id'] == flow_id][feature_columns].values
    pred2_data = pred2_df[pred2_df['flow_id'] == flow_id][feature_columns].values
    
    # 归一化到 0-255
    ini_data = normalize_and_scale(ini_data)
    pred1_data = normalize_and_scale(pred1_data)
    pred2_data = normalize_and_scale(pred2_data)

    # 跳过数据缺失的情况
    if ini_data is None or pred1_data is None or pred2_data is None:
        print(f"⚠️ Warning: flow_id {flow_id} has missing data, skipping...")
        continue

    # 组装成 RGB 三通道
    rgb_image = np.stack([ini_data, pred1_data, pred2_data], axis=-1)  # (32, 6, 3)

    # 将 (32,6,3) 转换为 (32,32,3) 以符合 CNN 输入格式
    rgb_image_resized = cv2.resize(rgb_image, (32, 32), interpolation=cv2.INTER_CUBIC)

    # 保存图片
    img_path = os.path.join(output_dir, f"{flow_id}.png")
    if cv2.imwrite(img_path, rgb_image_resized):
        generated_images += 1  # 统计成功生成的图片

print(f"✅ 任务完成，共生成 {generated_images} 张图片，存放于 {output_dir}/ 目录。")
