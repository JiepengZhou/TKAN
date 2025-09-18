import numpy as np
import torch
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from TKAN import TKAN
from DataProcess import get_train_data, get_valid_data, get_test_data, get_train_fit_data, get_normal_data

def autoregressive_forecast_with_pacc(model, padded_seq, pacc, target_length=32, feature_dim=1):
    while pacc < target_length:
        input_seq = np.copy(padded_seq)
        input_seq[pacc:, :] = 0  # 确保后面部分归零

        input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # (1, 32, feature_dim)
        output_seq = model(input_seq).detach().cpu().numpy()  # 确保数据在 CPU 上

        next_step = output_seq[:, pacc, :]  # (1, feature_dim)
        padded_seq[pacc] = next_step
        pacc += 1
    
    return padded_seq  # (32, feature_dim)

# 数据归一化
def get_normal_data(purchase_seq, target_features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_data = scaler.fit_transform(purchase_seq[target_features])
    print(f"scaled_data shape: {np.array(scaler_data).shape}")
    return scaler_data, scaler

# 数据反归一化
def inverse_transform(predictions, scaler):
    return scaler.inverse_transform(predictions)

target_columns = ['length', 'inter_arrival_time', 'rolling_mean_length', 'rolling_std_length', 'rolling_max_length', 'rolling_min_length']
rnn_unit = 64
feature_num = len(target_columns)
kan_layer_hidden = [feature_num + rnn_unit, 64, feature_num]  # 修正 `output_size` 变量未定义问题
time_step = 32

# 加载模型
model = TKAN(feature_num, rnn_unit, kan_layer_hidden, feature_num)
model.load_state_dict(torch.load('best_TKAN132.pth', map_location=torch.device('cpu')))
model.eval()

IDS_TYPE = sys.argv[1]
file_name = "TSF/" + IDS_TYPE + "wz.csv"
fdf = pd.read_csv(file_name)

# 归一化数据
InitialData, scaler = get_normal_data(fdf, target_columns)
InitialData = pd.DataFrame(InitialData, columns=target_columns)
InitialData['flow_id'] = fdf['flow_id'].values  # 需要确保 flow_id 仍然存在
InitialData['pacc'] = fdf['pacc'].values

final_results = []

# 按流 ID 分组
for flow_id, group in tqdm(InitialData.groupby('flow_id'), desc="Processing flow IDs", unit="flow"):
    pacc = int(group['pacc'].iloc[0])

    padded_seq = np.zeros((32, feature_num))
    valid_data = group[target_columns].values
    padded_seq[:pacc] = valid_data

    predicted_seq = autoregressive_forecast_with_pacc(model, padded_seq, pacc, target_length=32, feature_dim=feature_num)
    
    # 反归一化保留结果
    pred = inverse_transform(predicted_seq.reshape(-1, feature_num), scaler).reshape(32, feature_num)

    for i in range(32):
        row = [flow_id, i] + pred[i].tolist()
        final_results.append(row)

output_columns = ['flow_id', 'time_step'] + target_columns
result_df = pd.DataFrame(final_results, columns=output_columns)
result_df['label'] = IDS_TYPE
result_df.to_csv("Pred1/" + IDS_TYPE + "pred1.csv", index=False)

print("预测完成，结果已保存到 predicted_results.csv")
