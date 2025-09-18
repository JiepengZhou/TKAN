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
from Resnet import create_resnet18
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
ne = 30 # num_epochs
nc = 10 # type
nw = 36 # num_workers paralley cpu process
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_resnet18().to(device) # 定义分类模型
print(model)

criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化函数
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

# 数据变换
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# 数据加载
train_dataset = ImageDataset(root_dir=os.path.join(extract_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
valid_dataset = ImageDataset(root_dir=os.path.join(extract_dir, 'val'), transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size = bs, shuffle=False, num_workers=nw, pin_memory=True)
test_dataset = ImageDataset(root_dir=os.path.join(extract_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
print("DataLoading Success!")

# 路径用于保存最优模型
best_model_path = "best_model.pth"
best_accuracy = 0.0
train_losses = []  # 训练损失
valid_losses = []  # 验证损失
model.load_state_dict(torch.load(best_model_path))

# 训练循环
print("start train...")
for epoch in range(ne):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # print(f"image shape: {images.shape}")
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(images)
        # **计算 Mixup/CutMix 损失**
        loss = criterion(outputs, labels)

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