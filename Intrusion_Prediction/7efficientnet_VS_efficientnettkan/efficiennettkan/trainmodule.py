import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
from ImgDataset import ImageDataset
from torch.utils.data import DataLoader
from CTKAN import CTKAN
from FocalLoss import FocalLoss as focalloss
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 颜色变换 & 亮度调整
color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

# Mixup 实现（需要配合 CutMix）
def mixup_data(x, y, alpha=1.0):
    '''Mixup 数据增强'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()  # 随机排列索引
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# CutMix 实现
def cutmix_data(x, y, alpha=1.0):
    '''CutMix 数据增强'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()  # 重新排列索引
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # 交换 CutMix 区域
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# 计算 CutMix 交换区域
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def one_hot(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes).float()
    
# 设置随机种子
seed = 56  
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  
    
extract_dir = "../dataset/extracted" # dataset directory
bs = 64 # batch_size
ne = 10 # num_epochs
nc = 10 # type
nw = 36 # num_workers paralley cpu process
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CTKAN().to(device) # 定义分类模型
print(model)

class_weights = torch.tensor([1.0, 1.0, 1.0, 7.8, 2.6, 5.8, 1.0, 7.8, 2.0, 5.8]).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights) # 损失函数
criterion = focalloss(alpha=class_weights, gamma=3.0).to(device) # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 优化函数
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

# 数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# 数据加载
train_dataset = ImageDataset(root_dir=os.path.join(extract_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
valid_dataset = ImageDataset(root_dir=os.path.join(extract_dir, 'val'), transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size = bs, shuffle=True, num_workers=nw, pin_memory=True)
test_dataset = ImageDataset(root_dir=os.path.join(extract_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
print("DataLoading Success!")

# 路径用于保存最优模型
best_model_path = "best_model.pth"
best_accuracy = 0.0
train_losses = []  # 训练损失
valid_losses = []  # 验证损失
# 直接加载已经好的模型继续小类别微调
# model.load_state_dict(torch.load(best_model_path))

# 训练循环
print("start train...")
for epoch in range(ne):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        # **随机选择是否应用 Mixup 或 CutMix**
        p = random.random()
        if p < 0.1:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
        elif p < 0.2:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=0.2)
        else:
            labels_a, labels_b, lam = labels, labels, 1  # 不增强

        num_classes = nc  
        labels_a = one_hot(labels_a, num_classes)
        labels_b = one_hot(labels_b, num_classes)

        outputs = model(images)

        # **计算 Mixup/CutMix 损失**
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch + 1}/{ne}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
    train_losses.append(running_loss / len(train_loader))  # 记录训练损失
    
    # 验证
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_accuracy = 100 * correct / total
    print(f'Validation Loss: {valid_loss / len(valid_loader):.4f}, Accuracy: {valid_accuracy:.2f}%')
    valid_losses.append(valid_loss / len(valid_loader))  # 记录验证损失
    
    # 使用 ReduceLROnPlateau 调整学习率
    scheduler.step(valid_loss / len(valid_loader))
    
    # 保存最优模型
    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"-----Saved best model at epoch {epoch+1} with accuracy {best_accuracy:.2f}%-----")

# 加载最优模型
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)  # 转为概率
        _, preds = torch.max(probs, 1)  # 获取预测类别

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算分类报告（包含 Precision, Recall, F1-Score）
report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)], digits=4)
print("\nTest Set Performance:\n", report)

plt.figure(figsize=(10, 6))
plt.plot(range(ne), train_losses, label='Train Loss', color='blue')
plt.plot(range(ne), valid_losses, label='Validation Loss', color='red')
plt.title('Train and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss.png')