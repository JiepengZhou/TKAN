# XGBoost-TKAN-INI
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from TKAN import TKAN

# 类别对应的csv文件路径
category_files = {
    0: 'benign',
    1: 'dos',
    2: 'ddos',
    3: 'ftp',
    4: 'ssh',
    5: 'port',
    6: 'inf',
    7: 'wa',
    8: 'hb',
    9: 'bn'
}

all_features = []
all_labels = []

scaler = MinMaxScaler(feature_range=(0, 1))  # 定义归一化器

for label, typecsv in category_files.items():
    file_path = 'INI/' + typecsv + 'ini.csv'
    df = pd.read_csv(file_path)
    
    grouped = df.groupby('flow_id')
    print(f"{typecsv}中总的流数目： {len(grouped)}")
    
    features = []  # 确保features在每个流循环中重新初始化为空列表
    for flow_id, group in grouped:
        feature = group.iloc[:, 8:14].values  # 提取特征
        feature = scaler.fit_transform(feature)
        features.append(feature)
    
    # 将所有特征合并成一个大数组
    features = np.vstack(features)
    
    # 每32 * 32行构成一个样本，确保行数是32 * 32的倍数
    num_samples = features.shape[0] // (32)
    print(f"{typecsv}实际读出来的样本数目：{num_samples}") 
    
    features = features.reshape(num_samples, 32, 6)
    print(f"features.shape: {features.shape}")
    
    all_features.append(features)
    all_labels.append(np.full(shape=(num_samples,), fill_value=label))

# 合并所有类别
X = np.vstack(all_features)  # (总样本数, 32, 6)
print(f"X.shape:{X.shape}")
y = np.concatenate(all_labels)  # (总样本数,)
print(f"y.shape:{y.shape}")

# 加载模型
input_size = 6  # 每个时间步的特征数量
hidden_size = 32
output_size = 10  # 分类数
kan_layers_hidden = [32, 64, 32]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TKAN(input_size, hidden_size, kan_layers_hidden, output_size).to(device)
model.load_state_dict(torch.load('best_tkan_INI.pth', map_location=device))
print(model)
model.eval()  # 切换为评估模式

# 将X从numpy转换为PyTorch的张量
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

# 创建一个空列表来存储所有样本的kan_output
kan_outputs = []

# 使用DataLoader或批量处理（防止显存溢出）
batch_size = 64  # 根据显存大小调整
num_batches = (X_tensor.shape[0] + batch_size - 1) // batch_size

for i in range(num_batches):
    # 切片获取当前批次
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, X_tensor.shape[0])
    batch = X_tensor[start_idx:end_idx]

    # 前向传播，获取KAN输出
    with torch.no_grad():
        lstm_output, _ = model.lstm(batch)  # 获取LSTM的输出
        last_step = lstm_output[:, -1, :]   # 取LSTM最后时间步
        kan_output = model.kan(last_step)  # 通过KAN前向传播
        
        # 将结果添加到列表
        kan_outputs.append(kan_output.cpu().numpy())

# 合并所有批次的KAN输出
kan_outputs = np.vstack(kan_outputs)
# 添加一个新的维度
kan_outputs = kan_outputs[:, :, None]  
# 将KAN输出沿最后一个轴拼接到原始输入X
X_new = np.concatenate((X, kan_outputs), axis=2)  # (num_samples, 32, 7)
# 将新的输出贴进X，由机器学习方法决定这一列权重多大
X = X_new

print()
print(f"X.shape (after KAN): {X.shape}")
print(f"y shape: {y.shape}")

# 随机打乱
X, y = shuffle(X, y, random_state=42)

# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 32 * 7), y, test_size=0.2, random_state=42, stratify=y)
print()
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_test.shape: {y_test.shape}")
print(f"y_test.shape: {y_test.shape}")

# 训练
print()
print("开始训练...")
clf = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    scale_pos_weight=1,  # 默认为1，调整为较大值帮助少数类
    objective='multi:softmax',  # 多分类任务
    num_class=10,  # 根据类别数调整
    eval_metric='mlogloss',  # 评估指标
    use_label_encoder=False  # 防止XGBoost显示警告
)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))