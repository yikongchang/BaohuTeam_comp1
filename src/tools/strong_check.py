# 数据增强校验
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image  # Use PIL for image loading to get size easily
import numpy as np

# Define paths (adjust these to your actual dataset paths)
dataset_root = r'E:\fish\comp\data'  # e.g., '/home/user/dataset'
train_images_dir = os.path.join(dataset_root, 'train', 'images')
train_labels_dir = os.path.join(dataset_root, 'train', 'labels')


# Function to load labels
def load_labels(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
    return [[int(cls), float(x), float(y), float(w), float(h)] for cls, x, y, w, h in labels]


# Function to draw bounding boxes on image
def draw_bboxes(image_path, labels, ax, title):
    # Load image
    img = np.array(Image.open(image_path))
    h, w = img.shape[:2]

    # Display image
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

    # Draw each bbox
    for cls, x, y, ww, hh in labels:
        # Convert YOLO format (center_x, center_y, width, height) normalized to pixels
        bbox_x = (x - ww / 2) * w
        bbox_y = (y - hh / 2) * h
        bbox_w = ww * w
        bbox_h = hh * h

        # Create rectangle patch
        rect = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox_x, bbox_y - 5, f'Class: {cls}', color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))


# Example: Choose one original file and one augmentation to verify
# Adjust these filenames to match your actual files
original_image_file = '2022-06-10-13-59-34-14543_2048-1536.png'  # e.g., 'image001.jpg'
aug_suffix = 'rot_180'  # Choose one: 'flip_h', 'flip_v', 'rot_90_cw', 'rot_90_ccw', 'rot_180'

original_image_path = os.path.join(train_images_dir, original_image_file)
original_label_file = os.path.splitext(original_image_file)[0] + '.txt'
original_label_path = os.path.join(train_labels_dir, original_label_file)

aug_image_file = os.path.splitext(original_image_file)[0] + f'_{aug_suffix}' + os.path.splitext(original_image_file)[1]
aug_image_path = os.path.join(train_images_dir, aug_image_file)
aug_label_file = os.path.splitext(original_image_file)[0] + f'_{aug_suffix}.txt'
aug_label_path = os.path.join(train_labels_dir, aug_label_file)

# Load labels
original_labels = load_labels(original_label_path)
aug_labels = load_labels(aug_label_path)

# Create figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Draw original
draw_bboxes(original_image_path, original_labels, axs[0], 'Original')

# Draw augmented
draw_bboxes(aug_image_path, aug_labels, axs[1], f'Augmented ({aug_suffix})')

# Show the plot
plt.tight_layout()
plt.show()