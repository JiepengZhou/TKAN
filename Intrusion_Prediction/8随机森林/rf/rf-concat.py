# 随机森林-ConcatDataset
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 类别对应的csv文件路径
category_files = {
    0: 'benign.csv',
    1: 'dos.csv',
    2: 'ddos.csv',
    3: 'ftp.csv',
    4: 'ssh.csv',
    5: 'port.csv',
    6: 'inf.csv',
    7: 'wa.csv',
    8: 'hb.csv',
    9: 'bn.csv'
}

all_features = []
all_labels = []

scaler = MinMaxScaler(feature_range=(0, 1))  # 定义归一化器

for label, typecsv in category_files.items():
    file_path = 'Concat/' + typecsv
    df = pd.read_csv(file_path)
    # print(f"df_shape: {df.shape}")
    
    grouped = df.groupby('flow_id')
    print(f"{typecsv}中总的流数目： {len(grouped)}")
    
    features = []  # 确保features在每个流循环中重新初始化为空列表
    for flow_id, group in grouped:
        feature = group.iloc[:, 0:18].values  # 提取特征
        feature = scaler.fit_transform(feature)
        features.append(feature)
    
    # 将所有特征合并成一个大数组
    features = np.vstack(features)
    
    # 每32行构成一个样本，确保行数是32的倍数
    num_samples = features.shape[0] // 32
    print(f"{typecsv}实际读出来的样本数目：{num_samples}") 
    
    # print(f"features.shape: {features.shape}")
    features = features.reshape(num_samples, 32, 18)
    print(f"features.shape: {features.shape}")
    
    all_features.append(features)
    all_labels.append(np.full(shape=(num_samples,), fill_value=label))

# 合并所有类别
X = np.vstack(all_features)  # (总样本数, 32, 18)
print(f"X.shape:{X.shape}")
y = np.concatenate(all_labels)  # (总样本数,)
print(f"y.shape:{y.shape}")

# 随机打乱
X, y = shuffle(X, y, random_state=42)

# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 32 * 18), y, test_size=0.2, random_state=42, stratify=y)
print()
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_test.shape: {y_test.shape}")
print(f"y_test.shape: {y_test.shape}")

# 训练
print()
print("开始训练...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))