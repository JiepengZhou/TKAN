import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from E import PureEfficientnet
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
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 将三通道都进行标准化
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
        self.features = features  # 形状: (总样本数, 3, 32, 6)
        self.labels = labels      # 形状: (总样本数,)
        self.transform = transform
        
    def __getitem__(self, idx):
        # 获取矩阵数据和标签
        matrix_data = self.features[idx]
        label = self.labels[idx]

        matrix_data = np.transpose(matrix_data, (1, 2, 0))
        
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

# 假设三个文件夹路径分别为 'folder1/', 'folder2/', 'folder3/'
folders = ['INI/', 'Pred1/', 'Pred2/']

# 遍历文件夹
for folder in folders:
    folder_features = [] # 存储当前文件夹中所有样本
    folder_labels = []  # 存储当前文件夹的所有标签
    
    print(f"##################### {folder} #######################")
    print()
    
    for label, file in category_files.items():
        if folder == 'INI/' :
            file_path = folder + file + 'ini.csv'
            begin = 8
            end = 14
        elif folder == 'Pred1/':
            file_path = folder + file + 'pred1.csv'
            begin = 2
            end = 8
        else :
            file_path = folder + file + 'pred2.csv'
            begin = 2
            end = 8
        df = pd.read_csv(file_path)
        
        grouped = df.groupby('flow_id')
        print(f"{file}中总的流数目： {len(grouped)}")
        
        features = []  # 确保features在每个流循环中重新初始化为空列表
        for flow_id, group in grouped:
            feature = group.iloc[:, begin:end].values  # 提取特征
            feature = scaler.fit_transform(feature)
            features.append(feature)
        
        # 将所有特征合并成一个大数组
        features = np.vstack(features)
        
        # 每32行构成一个样本，确保行数是32的倍数
        num_samples = features.shape[0] // 32
        print(f"{file}实际读出来的样本数目：{num_samples}")
        
        features = features[:num_samples * 32].reshape(num_samples, 32, 6)
        print(f"features.shape: {features.shape}")
        
        folder_features.append(features)
        if folder == 'INI/':  # 只保存第一个文件夹的标签
            all_labels.append(np.full(shape=(num_samples,), fill_value=label))
    
    folder_features = np.vstack(folder_features) # 将当前文件夹的所有样本堆叠到一个数组中，形成 (num_samples, 3, 32, 6)
    print(f"============={folder}文件夹中所有样本的 shape: {folder_features.shape}=============")
    
    # 将文件夹的特征数组添加到总特征列表中
    all_features.append(folder_features)
    
# 合并三个文件夹的数据，形成 (num_samples, 3, 32, 6)
X = np.stack(all_features, axis=1)  # 在通道维度堆叠
print(f"X.shape: {X.shape}")
y = np.concatenate(all_labels)  # (总样本数,)
print(f"y.shape:{y.shape}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 创建数据集和 DataLoader
train_dataset = FlowDataset(X_train, y_train, transform=custom_transform)
val_dataset = FlowDataset(X_val, y_val, transform=custom_transform)
test_dataset = FlowDataset(X_test, y_test, transform=custom_transform)

bs = 64
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)

# 打印 DataLoader 的信息
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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
best_val_loss = float('inf')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model_path = 'best_e_cat.pth'
model = PureEfficientnet(num_classes=10).to(device)  # 假设你有 10 个类别
# model.load_state_dict(torch.load("best_e_cat.pth"))
print(model)

criterion = nn.CrossEntropyLoss()  # 对于多分类问题，使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Early Stopping 参数
early_stop_patience = 3  # 如果超过3个epoch验证集准确率未提升就停止
no_improvement_count = 0  # 记录连续未提升的epoch数量

for epoch in range(num_epochs):
    print(f"| Epoch {epoch+1}/{num_epochs}")

    # 训练一个epoch
    train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"| Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}% |")
    
    # 验证一个epoch
    val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
    print(f"| Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}% |")
    
    scheduler.step(val_loss)
    
    # 在准确率上升时保留最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(">>>>>>>>>>>>>Best model saved!>>>>>>>>>>>>>>>>")

    # 在验证集损失上升3次时早停避免模型牺牲置信度来获得高准确率，降低泛化能力
    # 其中在损失上升2次后会降低学习率，若接下来还是上升，则有理由认为模型开始过拟合
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_count = 0  # 重置计数器
    else:
        no_improvement_count += 1  # 未提升计数器加1
    
    # 检查是否需要早停
    if no_improvement_count >= early_stop_patience:
        print(f"Early stopping triggered after {epoch+1} epochs!")
        break

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
print(classification_report(all_labels, all_preds, labels=list(label_map.keys()), target_names=list(label_map.values()), digits=4))