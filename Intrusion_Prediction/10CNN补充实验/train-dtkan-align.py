import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from DTKAN import CTKAN
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

# 定义归一化器
scaler = MinMaxScaler(feature_range=(0, 1)) 

# 数据预处理的 transform
custom_transform = transforms.Compose([
    transforms.ToPILImage(),                          # 转换为 PIL 图像
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),  # 双三次插值
    transforms.ToTensor(),                            # 转为 Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])       # 标准化
])

# 标签字典映射
label_map = {
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

# Dataset 类
class FlowDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features  # 形状: (总样本数, 32, 18)
        self.labels = labels      # 形状: (总样本数,)
        self.transform = transform
        
    def __getitem__(self, idx):
        # 获取矩阵数据和标签
        matrix_data = self.features[idx]
        label = self.labels[idx]

        # 扩展为 1 通道的图像格式 (32, 18, 1)
        matrix_data = np.expand_dims(matrix_data, axis=-1)

        # 数据先进行了归一化，再映射到0-255，最后还得将float64转为uint8
        matrix_data = np.clip(matrix_data * 255, 0, 255).astype(np.uint8)

        # 应用 transform
        if self.transform:
            matrix_data = self.transform(matrix_data)

        return matrix_data, label

    def __len__(self):
        return len(self.features)

# 读取数据并处理
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

for label, typecsv in category_files.items():
    file_path = 'Concat/' + typecsv
    df = pd.read_csv(file_path)
    
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
    num_samples = features.shape[0] // (32)
    print(f"{typecsv}实际读出来的样本数目：{num_samples}") 
    
    features = features.reshape(num_samples, 32, 18)
    print(f"features.shape: {features.shape}")
    
    all_features.append(features)
    all_labels.append(np.full(shape=(num_samples,), fill_value=label))

# 合并所有类别
X = np.vstack(all_features)  # (总样本数, 32, 18)
print(f"X.shape:{X.shape}")
y = np.concatenate(all_labels)  # (总样本数,)
print(f"y.shape:{y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_dataset = FlowDataset(X_train, y_train, transform=custom_transform)
test_dataset = FlowDataset(X_test, y_test, transform=custom_transform)

bs = 64
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)

# 打印训练集和测试集的数据量
print(f"训练集 DataLoader 的大小: {len(train_loader.dataset)}")
print(f"测试集 DataLoader 的大小: {len(test_loader.dataset)}")

# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累加损失
        running_loss += loss.item()
        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 验证函数
def validate_epoch(model, test_loader, criterion, device):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in tqdm(test_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 累加损失
            running_loss += loss.item()
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 训练和验证循环
num_epochs = 30 
best_val_accuracy = 0.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model_path = 'best_dtkan_align.pth'
model = CTKAN(num_classes=10).to(device)  # 假设你有 10 个类别
print(model)
criterion = nn.CrossEntropyLoss()  # 对于多分类问题，使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

for epoch in range(num_epochs):
    print(f"| Epoch {epoch+1}/{num_epochs}")

    # 训练一个epoch
    train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"| Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}% |")
    
    # 验证一个epoch
    val_loss, val_accuracy = validate_epoch(model, test_loader, criterion, device)
    print(f"| Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}% |")
    
    scheduler.step(val_loss)
    
    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(">>>>>>>>>>>>>Best model saved!>>>>>>>>>>>>>>>>")

print("Training finished!")

# 模型切换为评估模式
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 打印分类报告
print("Classification Report on Test Set:")
print(classification_report(all_labels, all_preds, digits=4))