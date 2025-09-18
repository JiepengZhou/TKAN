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

extract_dir = "../dataset/extracted" # dataset directory
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bs = 64
nw = 18
model = CTKAN().to(device) # 定义分类模型
print(model)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 确保输入尺寸匹配 ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet 预训练模型的标准化参数
])

test_dataset = ImageDataset(root_dir=os.path.join(extract_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
print("DataLoading Success!")

# 加载最优模型
best_model_path = "best_model.pth"
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

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true, y_pred = [], []
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("ConfusionMatrix.png")


'''

# ---------------------------------
# **Grad-CAM 可视化**
# ---------------------------------
def generate_gradcam(model, image, label, target_layer):
    # 保存原始模式
    is_training = model.training
    model.train()  # 将模型切换到训练模式，确保RNN可以进行反向传播

    image = image.unsqueeze(0).to(device)

    # 前向传播
    features = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal features
        features = out

    def backward_hook(module, grad_inp, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # 注册 hook
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # 计算输出
    output = model(image)
    class_idx = torch.argmax(output, dim=1).item()

    model.zero_grad()
    output[:, class_idx].backward()

    # 计算 Grad-CAM
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * features, dim=1)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().detach().numpy()

    # 归一化
    cam = cv2.resize(cam, (224, 224))
    cam_min = np.min(cam)
    cam_max = np.max(cam)
    if cam_max != cam_min:  # 避免除以零
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)  # 如果最大值和最小值相同，则设置为全零

    # 叠加到原图上
    img = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = (img * 0.225 + 0.45)  # 反标准化
    img = np.clip(img, 0, 1)

    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    result = heatmap * 0.4 + img * 0.6

    handle_forward.remove()
    handle_backward.remove()

    # 恢复为原始模式
    if not is_training:
        model.eval()

    return result, label, class_idx


# 选择 ResNet18 的最后一个卷积层（layer4）
target_layer = model.resnet.resnet[-1]

# 选取一些测试图像进行 Grad-CAM 可视化
for idx in range(10):
    print(f"idx: {idx}")
    image, label = test_dataset[idx]
    cam_img, true_label, pred_label = generate_gradcam(model, image, label, target_layer)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(cam_img)
    plt.title(f"True: {true_label}, Pred: {pred_label}")
    plt.axis("off")
    plt.savefig(f"gradcam_class_{true_label}_idx_{idx}.png")

print("Grad-CAM visualization saved for each class!")
'''