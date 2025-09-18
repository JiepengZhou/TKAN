import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import random
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
from ImgDataset import ImageDataset
from torch.utils.data import DataLoader
from CTKAN import CTKAN
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.decomposition import PCA

# 获取特征和标签
def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.cpu().numpy()
            
            # 提取 ResNet 特征
            feature = model.resnet(images)
            feature = feature.view(feature.size(0), -1).cpu().numpy()
            
            features.append(feature)
            labels.append(targets)
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# 可视化 PCA 结果
def visualize_pca(features, labels, num_classes=10):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        idxs = labels == i
        plt.scatter(reduced_features[idxs, 0], reduced_features[idxs, 1], label=f"Class {i}", alpha=0.5)
    
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.title("PCA Visualization of Extracted Features")
    plt.savefig("PCA.png")

# 加载数据集
extract_dir = "../dataset/extracted"
batch_size = 64
bs = 64
nw = 16

# 数据变换
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 随机裁剪
    transforms.RandomRotation(degrees=15),  # 随机旋转
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 颜色增强
    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageDataset(root_dir=os.path.join(extract_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载最优模型
best_model_path = "best_model.pth"
model = CTKAN().to(device)
model.load_state_dict(torch.load(best_model_path))

# 提取特征
features, labels = extract_features(model, test_loader, device)

# 进行 PCA 可视化
visualize_pca(features, labels)
