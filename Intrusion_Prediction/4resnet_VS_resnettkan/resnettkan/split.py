import os
import tarfile
import random
import shutil

def extract_and_split_data(data_dir, extract_dir, train_ratio=0.7, val_ratio=0.1):
    os.makedirs(extract_dir, exist_ok=True)

    # 解压 tar.gz 文件
    for tar_file in os.listdir(data_dir):
        if tar_file.endswith(".tar.gz"):
            class_name = tar_file.split(".tar.gz")[0]
            class_extract_path = os.path.join(extract_dir, class_name)
            os.makedirs(class_extract_path, exist_ok=True)

            with tarfile.open(os.path.join(data_dir, tar_file), 'r:gz') as tar:
                tar.extractall(class_extract_path)
                print(f"Extracted {tar_file} to {class_extract_path}")
                print("Files after extraction:", os.listdir(class_extract_path))

    # 创建 train, val, test 目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(extract_dir, split), exist_ok=True)

    for class_name in os.listdir(extract_dir):
        class_path = os.path.join(extract_dir, class_name)
        if os.path.isdir(class_path) and class_name not in ['train', 'val', 'test']:
            # **进入子目录** 获取所有图片
            subdirs = os.listdir(class_path)
            if len(subdirs) == 1 and os.path.isdir(os.path.join(class_path, subdirs[0])):
                class_path = os.path.join(class_path, subdirs[0])  # 进入真正的图片目录

            images = os.listdir(class_path)
            print(f"Class: {class_name}, Total images found: {len(images)}")

            if not images:  # 如果没有图片，跳过
                continue

            random.shuffle(images)

            train_idx = int(len(images) * train_ratio)
            val_idx = int(len(images) * (train_ratio + val_ratio))

            for split, img_list in zip(['train', 'val', 'test'], 
                                       [images[:train_idx], images[train_idx:val_idx], images[val_idx:]]):
                split_path = os.path.join(extract_dir, split, class_name)
                os.makedirs(split_path, exist_ok=True)
                for img in img_list:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(split_path, img)
                    print(f"Moving {src} -> {dst}")
                    shutil.move(src, dst)

            try:
                os.rmdir(class_path)
                print(f"Deleted empty folder: {class_path}")
            except OSError:
                print(f"Could not delete {class_path}, not empty")

# 运行代码
extract_and_split_data('dataset/tar_files', 'dataset/extracted')
