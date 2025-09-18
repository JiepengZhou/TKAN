import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MyKan import KAN
from DataProcess import get_train_data, get_valid_data, get_test_data, get_train_fit_data, get_normal_data
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 模型参数
target_columns = ['length', 'inter_arrival_time', 'rolling_mean_length','rolling_std_length','rolling_max_length','rolling_min_length']
torch.manual_seed(42)  # 设置 PyTorch 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)  # 设置所有 GPU 的随机种子
lr = 0.0005  # 学习率
batch_size = 32  # mini-batch 大小
input_size = len(target_columns)  # 输入特征数量
output_size = len(target_columns)  # 输出特征数量
feature_num = len(target_columns)
kan_layer_hidden = [feature_num, 64, feature_num] # kan的维度，因为数据从LSTM出来是隐藏层的维度
time_step = 32  # 时间步长
epochs = 100  # 训练次数
gradient_threshold = 5.0  # 梯度裁剪阈值
stop_loss = 0.000000025  # 训练停止条件
train_keep_prob = [1.0, 0.5, 1.0]  # Dropout 保留率

# 数据准备
print("Loading Data ...")
purchase_seq = pd.read_csv('first_packet_processed.csv', usecols=['timestamp'] + target_columns, index_col='timestamp')
print(purchase_seq.head(10))

# 数据切分参数
divide_train_valid_index = int(len(purchase_seq) * 0.7)
divide_valid_test_index = int(divide_train_valid_index + int(len(purchase_seq) * 0.1))

# 归一化
scaled_x_data, scaled_y_data, scaler_y = get_normal_data(purchase_seq, target_columns)

# 数据划分
train_x, train_y = get_train_data(scaled_x_data, scaled_y_data, divide_train_valid_index, time_step)
train_fit_x, train_fit_y, train_fit_remain = get_train_fit_data(scaled_x_data, scaled_y_data, divide_train_valid_index, time_step)
valid_x, valid_y, valid_remain = get_valid_data(scaled_x_data, scaled_y_data, divide_train_valid_index, divide_valid_test_index, time_step)
test_x, test_y, test_remain = get_test_data(scaled_x_data, scaled_y_data, divide_valid_test_index, time_step)

# 转换为 PyTorch 张量
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
train_fit_x = torch.tensor(train_fit_x, dtype=torch.float32)
train_fit_y = torch.tensor(train_fit_y, dtype=torch.float32)
valid_x = torch.tensor(valid_x, dtype=torch.float32)
valid_y = torch.tensor(valid_y, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

# 检查是否有异常值
print(f"Train_X_HasNAN: {torch.isnan(train_x).any()}, Train_X_HasINF: {torch.isinf(train_x).any()}")
print(f"Train_Y_HasNAN: {torch.isnan(train_y).any()}, Train_Y_HasINF: {torch.isinf(train_y).any()}")
print(f"Train_Fit_X_HasNAN: {torch.isnan(train_fit_x).any()}, Train_Fit_X_HasINF: {torch.isinf(train_fit_x).any()}")
print(f"Train_Fit_Y_HasNAN: {torch.isnan(train_fit_y).any()}, Train_Fit_Y_HasINF: {torch.isinf(train_fit_y).any()}")
print(f"Valid_X_HasNAN: {torch.isnan(valid_x).any()}, Valid_X_HasINF: {torch.isinf(valid_x).any()}")
print(f"Valid_Y_HasNAN: {torch.isnan(valid_y).any()}, Valid_Y_HasINF: {torch.isinf(valid_y).any()}")
print(f"Test_X_HasNAN: {torch.isnan(test_x).any()}, Test_X_HasINF: {torch.isinf(test_x).any()}")
print(f"Test_Y_HasNAN: {torch.isnan(test_y).any()}, Test_Y_HasINF: {torch.isinf(test_y).any()}")

# 初始化模型、损失函数和优化器
model = KAN(kan_layer_hidden)
criterion = nn.MSELoss()  # 多输出均方误差损失
optimizer = optim.Adam(model.parameters(), lr=lr)

# 将数据模型搬到gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_x = train_x.to(device)
# train_y = train_y.to(device)
train_fit_x = train_fit_x.to(device)
train_fit_y = train_fit_y.to(device)
valid_x = valid_x.to(device)
valid_y = valid_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)
model = model.to(device)

# 训练函数
def train_kan():
    print("Starting Train...")
    fit_loss_seq = []
    valid_loss_seq = []
    best_valid_loss = float('inf')  # 记录最佳验证损失
    best_model_state = None         # 记录最佳模型

    for epoch in range(epochs):
        torch.cuda.empty_cache() 
        model.train()
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i + batch_size].to(device)
            batch_y = train_y[i:i + batch_size].to(device)
            # print(f"batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}")
            optimizer.zero_grad()
            output = model(batch_x)
            # print(f"output shape: {output.shape}, batch_y.shape: {batch_y.shape}")
            # print(f"output_HasNAN: {torch.isnan(output).any()}, output_HasINF: {torch.isinf(output).any()}")
            # print(f"batch_y_HasNAN: {torch.isnan(batch_y).any()}, batch_y_HasINF: {torch.isinf(batch_y).any()}")
            loss = criterion(output, batch_y)
            # print(f"loss: {loss}, Model Successfully processed!")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_threshold)
            optimizer.step()

        # 验证集评估
        model.eval()
        with torch.no_grad():
            train_fit_output = model(train_fit_x)
            train_loss = criterion(train_fit_output, train_fit_y)
            fit_loss_seq.append(train_loss.item())

            valid_output = model(valid_x)
            valid_loss = criterion(valid_output, valid_y)
            valid_loss_seq.append(valid_loss.item())

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss.item():.8f}, Valid Loss: {valid_loss.item():.8f}')

        # 记录最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict().copy()
            print(f'New best model saved at epoch {epoch + 1}')
            
        # 提前停止条件
        if train_loss + valid_loss <= stop_loss:
            print(f'Early stopping at epoch {epoch + 1}')
            break
     # 训练结束后保存最佳模型
    torch.save(best_model_state, 'best_KAN32.pth')
    print("Best model saved as best_KAN32.pth")
    return fit_loss_seq, valid_loss_seq

# 训练模型
fit_loss_seq, valid_loss_seq = train_kan()

# 可视化损失
plt.plot(fit_loss_seq, label='Train Loss')
plt.plot(valid_loss_seq, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losskan32.png')

# 反归一化预测结果
def inverse_transform(predictions, scaler):
    return scaler.inverse_transform(predictions)

# 测试集预测
print("Test ...")
best_model = model  # 重新加载模型
best_model.load_state_dict(torch.load('best_KAN32.pth'))
best_model.eval()

with torch.no_grad():
    test_output = best_model(test_x)
    test_output = test_output.cpu().numpy()
    test_output = inverse_transform(test_output.reshape(-1, feature_num), scaler_y)  # 反归一化


# 计算 MSE 和 MAE
def cal_mse_mae(pred, real, target_columns):
    n_features = real.shape[1]
    mse_per_feature = np.zeros(n_features)
    mae_per_feature = np.zeros(n_features)

    for i in range(n_features):
        mse_per_feature[i] = np.sqrt(mean_squared_error(real[:, i], pred[:, i]))
        mae_per_feature[i] = mean_absolute_error(real[:, i], pred[:, i])

    for i in range(n_features):
        print(f"Feature {target_columns[i]}: MSE = {mse_per_feature[i]:.4f}, MAE = {mae_per_feature[i]:.4f}")


# 如果 test_y 是 GPU 上的张量，转换为 NumPy 数组
test_y = test_y.cpu().numpy()
test_y = inverse_transform(test_y.reshape(-1, feature_num), scaler_y)  # 反归一化

print(f"test_y shape: {test_y.shape}, test_output shape: {test_output.shape}")
cal_mse_mae(test_output, test_y, target_columns)