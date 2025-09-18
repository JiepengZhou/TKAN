import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # 读取数据
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # 确定难分类类别
        self.hard_classes = [3, 4, 7] 
        self.very_hard_classes = [5, 9]
        
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            img_paths = [os.path.join(cls_dir, img_name) for img_name in os.listdir(cls_dir)]
    
            # 若类别是难分类类别，则复制数据
            if self.class_to_idx[cls_name] in self.very_hard_classes:
                img_paths = img_paths * 12  # 变成12倍数据量
            elif self.class_to_idx[cls_name] in self.hard_classes:
                img_paths = img_paths * 9 # 变成9倍数据量
            else :
                img_paths = img_paths * 5 # 其他变成5倍数据量
    
            self.data.extend(img_paths)
            self.labels.extend([self.class_to_idx[cls_name]] * len(img_paths))

        # 额外的增强变换
        self.color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.erasing_transform = transforms.RandomErasing(p=1.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 只对难分类类别做额外增强
        if label in self.hard_classes:
            if random.random() < 0.3:
                image = self.color_transform(image)
            if random.random() < 0.3:
                image = self.erasing_transform(image)

        return image, label

    def print_class_mapping(self):
        print("类别索引 → 真实类别名称 对应关系:")
        for i, cls in enumerate(self.classes):
            print(f"{i} → {cls}")
