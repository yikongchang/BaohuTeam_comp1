import os
import shutil
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
import albumentations as A

# ===================== 数据划分 =====================
dataset_root = r'E:\fish\comp\origin'
train_images_dir = os.path.join(dataset_root, 'train', 'images')
train_labels_dir = os.path.join(dataset_root, 'train', 'labels')
valid_images_dir = os.path.join(dataset_root, 'valid', 'images')
valid_labels_dir = os.path.join(dataset_root, 'valid', 'labels')

os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(valid_labels_dir, exist_ok=True)

image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

train_size = int(len(image_files) * 0.8)
train_files = image_files[:train_size]
valid_files = image_files[train_size:]

for file in valid_files:
    src_img = os.path.join(train_images_dir, file)
    dst_img = os.path.join(valid_images_dir, file)
    shutil.move(src_img, dst_img)

    label_file = os.path.splitext(file)[0] + '.txt'
    src_label = os.path.join(train_labels_dir, label_file)
    if os.path.exists(src_label):
        dst_label = os.path.join(valid_labels_dir, label_file)
        shutil.move(src_label, dst_label)

print(f"划分完成: {len(train_files)} train, {len(valid_files)} valid")


# ===================== 工具函数 =====================
def load_labels(label_path):
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, w, h = line.strip().split()
            labels.append([int(cls), float(x), float(y), float(w), float(h)])
    return labels


def save_labels(label_path, labels):
    with open(label_path, 'w') as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_albumentations(labels, img_w, img_h):
    bboxes = []
    for cls, x, y, w, h in labels:
        x_min = (x - w/2) * img_w
        y_min = (y - h/2) * img_h
        x_max = (x + w/2) * img_w
        y_max = (y + h/2) * img_h
        bboxes.append([x_min, y_min, x_max, y_max, cls])
    return bboxes


def albumentations_to_yolo(bboxes, img_w, img_h):
    labels = []
    for x_min, y_min, x_max, y_max, cls in bboxes:
        x = ((x_min + x_max) / 2) / img_w
        y = ((y_min + y_max) / 2) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        labels.append([int(cls), x, y, w, h])
    return labels


# ===================== 增强策略 =====================
transform = A.Compose([
    # 几何
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),

    # 光学
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.MotionBlur(p=0.2),
    A.HueSaturationValue(p=0.3),

    # 高级（mixup/cutmix通常在dataloader中做，这里简单加个模仿）
    A.CoarseDropout(num_holes=5, max_h_size=30, max_w_size=30, p=0.3)

], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# ===================== 执行增强 =====================
train_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for file in train_files:
    img_path = os.path.join(train_images_dir, file)
    label_path = os.path.join(train_labels_dir, os.path.splitext(file)[0] + '.txt')

    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    labels = load_labels(label_path)
    if not labels:
        continue

    bboxes = yolo_to_albumentations(labels, w, h)
    class_labels = [cls for _, _, _, _, cls in bboxes]

    # 做3个增强版本
    for i in range(3):
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_cls = transformed['class_labels']

        new_labels = albumentations_to_yolo(
            [[*box[:4], cls] for box, cls in zip(aug_bboxes, aug_cls)],
            aug_img.shape[1], aug_img.shape[0]
        )

        # 保存
        base_name, ext = os.path.splitext(file)
        new_img_name = f"{base_name}_aug{i}{ext}"
        new_label_name = f"{base_name}_aug{i}.txt"

        cv2.imwrite(os.path.join(train_images_dir, new_img_name), aug_img)
        save_labels(os.path.join(train_labels_dir, new_label_name), new_labels)

print("增强完成: 几何+光学+Cutout 已生成")
