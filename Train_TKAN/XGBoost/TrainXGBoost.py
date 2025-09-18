import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 读取数据
target_columns = ['length', 'inter_arrival_time', 'rolling_mean_length', 'rolling_std_length', 'rolling_max_length',
                  'rolling_min_length']
purchase_seq = pd.read_csv('first_packet_processed.csv', usecols=['timestamp'] + target_columns, index_col='timestamp')

# 数据切分
divide_train_valid_index = int(len(purchase_seq) * 0.7)
divide_valid_test_index = int(divide_train_valid_index + int(len(purchase_seq) * 0.1))

# 归一化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaled_x_data = scaler_x.fit_transform(purchase_seq[target_columns])
scaled_y_data = scaler_y.fit_transform(purchase_seq[target_columns])  # 目标变量也是时间序列数据

# 数据集切分
train_x, train_y = scaled_x_data[:divide_train_valid_index], scaled_y_data[:divide_train_valid_index]
valid_x, valid_y = scaled_x_data[divide_train_valid_index:divide_valid_test_index], scaled_y_data[divide_train_valid_index:divide_valid_test_index]
test_x, test_y = scaled_x_data[divide_valid_test_index:], scaled_y_data[divide_valid_test_index:]

# XGBoost 训练
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.01, max_depth=6, eval_metric="rmse")

# 训练时监控损失
evals_result = {}
xgb_model.fit(
    train_x, train_y,
    eval_set=[(train_x, train_y), (valid_x, valid_y)],
    verbose=True
)

# 获取训练过程中的损失
evals_result = xgb_model.evals_result()
train_loss = evals_result['validation_0']['rmse']
valid_loss = evals_result['validation_1']['rmse']

# 画损失曲线
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(valid_loss, label='Valid Loss', color='green')
plt.xlabel('Iterations')
plt.ylabel('RMSE Loss')
plt.legend()
plt.title('XGBoost Training and Validation Loss')
plt.savefig('loss_of_xgboost.png')
plt.show()

# **在测试集上进行预测**
test_pred = xgb_model.predict(test_x)

# **反归一化预测结果**
test_pred = scaler_y.inverse_transform(test_pred.reshape(-1, len(target_columns)))
test_y_original = scaler_y.inverse_transform(test_y)

# **对每个特征单独计算 MSE 和 MAE**
mse_values = []
mae_values = []

for i, feature in enumerate(target_columns):
    mse = mean_squared_error(test_y_original[:, i], test_pred[:, i])
    mse = np.sqrt(mse)
    mae = mean_absolute_error(test_y_original[:, i], test_pred[:, i])
    mse_values.append(mse)
    mae_values.append(mae)
    print(f"Feature: {feature}, Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

    # **画每个特征的预测 vs 真实值曲线**
    plt.figure(figsize=(10, 6))
    plt.plot(test_y_original[:, i], label='True', color='blue')
    plt.plot(test_pred[:, i], label='Predicted', color='red', linestyle='dashed')
    plt.xlabel("Samples")
    plt.ylabel(feature)
    plt.legend()
    plt.title(f'XGBoost Prediction vs True ({feature})')
    plt.savefig(f'XGBOOST/xgboost_test_prediction_{feature}.png')