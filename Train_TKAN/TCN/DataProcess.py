import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 数据归一化
def get_normal_data(purchase_seq, target_features):
    # 初始化归一化器
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_data = scaler.fit_transform(purchase_seq[target_features])
    # 目标变量和特征都归一化
    scaled_x_data = scaler_data
    scaled_y_data = scaler_data
    # 返回归一化后的特征数据、目标数据和归一化器
    # 维度：（r：样本数，c: 特征数）
    print(f"scaled_x_data shape: {np.array(scaled_x_data).shape}, scaled_y_data shape: {np.array(scaled_y_data).shape}")
    return scaled_x_data, scaled_y_data, scaler

# 获得训练数据
def get_train_data(scaled_x_data, scaled_y_data, divide_train_valid_index, time_step):
    train_x, train_y = [], []
    normalized_train_feature = scaled_x_data[0: divide_train_valid_index]
    normalized_train_label = scaled_y_data[0: divide_train_valid_index]
    for i in range(len(normalized_train_feature) - time_step + 1):
        train_x.append(normalized_train_feature[i:i + time_step].tolist())
        train_y.append(normalized_train_label[i:i + time_step].tolist())
    # train_x的维度（z:总的有多少个滑动窗口，r：时间步time_step，c:特征数）
    # train_y的维度（z:总的有多少个滑动窗口，r：时间步time_step，c:特征数）
    print(f"train_x shape: {np.array(train_x).shape}, train_y shape: {np.array(train_y).shape}")
    return train_x, train_y

# 获得训练拟合数据
def get_train_fit_data(scaled_x_data, scaled_y_data, divide_train_valid_index, time_step):
    train_fit_x, train_fit_y = [], []
    normalized_train_feature = scaled_x_data[0: divide_train_valid_index]
    normalized_train_label = scaled_y_data[0: divide_train_valid_index]
    train_fit_remain = len(normalized_train_label) % time_step
    train_fit_num = int((len(normalized_train_label) - train_fit_remain) / time_step)
    for i in range(train_fit_num):
        # 特征数据
        train_fit_x.append(normalized_train_feature[i * time_step:(i + 1) * time_step].tolist())
        # 目标数据
        train_fit_y.append(normalized_train_label[i * time_step:(i + 1) * time_step].tolist())
    if train_fit_remain > 0:
        # 处理剩余数据
        train_fit_x.append(normalized_train_feature[-time_step:].tolist())
        train_fit_y.append(normalized_train_label[-time_step:].tolist())
    print(f"train_fit_x shape: {np.array(train_fit_x).shape}, train_fit_y shape: {np.array(train_fit_y).shape}")
    # train_fit_x的维度（z:滑动窗口的数量，r:时间步timestep，c:特征数）
    # train_fit_y的维度（z:滑动窗口的数量，r:时间步timestep，c:特征数）
    return train_fit_x, train_fit_y, train_fit_remain

# 获得验证数据
def get_valid_data(scaled_x_data, scaled_y_data, divide_train_valid_index, divide_valid_test_index, time_step):
    valid_x, valid_y = [], []
    normalized_valid_feature = scaled_x_data[divide_train_valid_index: divide_valid_test_index]
    normalized_valid_label = scaled_y_data[divide_train_valid_index: divide_valid_test_index]
    valid_remain = len(normalized_valid_label) % time_step
    valid_num = int((len(normalized_valid_label) - valid_remain) / time_step)
    for i in range(valid_num):
        valid_x.append(normalized_valid_feature[i * time_step:(i + 1) * time_step].tolist())
        valid_y.append(normalized_valid_label[i * time_step:(i + 1) * time_step].tolist())
    if valid_remain > 0:
        valid_x.append(normalized_valid_feature[-time_step:].tolist())
        valid_y.append(normalized_valid_label[-time_step:].tolist())
    print(f"valid_x shape: {np.array(valid_x).shape}, valid_y shape: {np.array(valid_y).shape}")
    # valid_x的维度（z:滑动窗口的数量，r:时间步timestep，c:特征数）
    # valid_y的维度（z:滑动窗口的数量，r:时间步timestep，c:特征数）
    return valid_x, valid_y, valid_remain

# 获得测试数据
def get_test_data(scaled_x_data, scaled_y_data, divide_valid_test_index, time_step):
    test_x, test_y = [], []
    normalized_test_feature = scaled_x_data[divide_valid_test_index:]
    normalized_test_label = scaled_y_data[divide_valid_test_index:]
    test_remain = len(normalized_test_label) % time_step
    test_num = int((len(normalized_test_label) - test_remain) / time_step)
    for i in range(test_num):
        test_x.append(normalized_test_feature[i * time_step:(i + 1) * time_step].tolist())
        test_y.append(normalized_test_label[i * time_step:(i + 1) * time_step].tolist())
    if test_remain > 0:
        test_x.append(scaled_x_data[-time_step:].tolist())
        test_y.append(normalized_test_label[-time_step:].tolist())
    print(f"test_x shape: {np.array(test_x).shape}, test_y shape: {np.array(test_y).shape}")
    # test_x的维度（z:滑动窗口的数量，r:时间步timestep，c:特征数）
    # test_y的维度（z:滑动窗口的数量，r:时间步timestep，c:特征数）
    return test_x, test_y, test_remain



