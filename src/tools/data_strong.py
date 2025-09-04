import os
import shutil
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate

# Define paths (adjust these to your actual dataset paths)
dataset_root = r'E:\fish\comp\data'  # e.g., '/home/user/dataset'
train_images_dir = os.path.join(dataset_root, 'train', 'images')
train_labels_dir = os.path.join(dataset_root, 'train', 'labels')
valid_images_dir = os.path.join(dataset_root, 'valid', 'images')
valid_labels_dir = os.path.join(dataset_root, 'valid', 'labels')

# Create valid directories if they don't exist
os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(valid_labels_dir, exist_ok=True)

# Step 1: Split train into train and valid (8:2 ratio)
# Get list of all image files in train/images
image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
total_images = len(image_files)


# Calculate split sizes
train_size = int(total_images * 0.9)
valid_size = total_images - train_size

# Shuffle and split
random.shuffle(image_files)
valid_files = image_files[:valid_size]
train_files = image_files[valid_size:]

# Move valid files to valid directory
for file in valid_files:
    # Move image
    src_img = os.path.join(train_images_dir, file)
    dst_img = os.path.join(valid_images_dir, file)
    shutil.move(src_img, dst_img)

    # Move corresponding label
    label_file = os.path.splitext(file)[0] + '.txt'
    src_label = os.path.join(train_labels_dir, label_file)
    if os.path.exists(src_label):
        dst_label = os.path.join(valid_labels_dir, label_file)
        shutil.move(src_label, dst_label)
    else:
        print(f"Warning: Label file {label_file} not found for {file}")

print(f"Moved {len(valid_files)} images and labels to valid.")


# Step 2: Data augmentation on remaining train
# Augmentations: horizontal flip, vertical flip, 90° clockwise, 90° counter-clockwise, 180° (upside down)

def load_labels(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
    return [[int(cls), float(x), float(y), float(w), float(h)] for cls, x, y, w, h in labels]


def save_labels(label_path, labels):
    with open(label_path, 'w') as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def augment_image_and_labels(image_path, label_path, aug_type, output_suffix):
    # Load image as numpy array
    image = plt.imread(image_path)
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    labels = load_labels(label_path)
    new_labels = []
    h, w = image.shape[:2]

    if aug_type == 'flip_horizontal':
        new_image = np.fliplr(image)
        for cls, x, y, ww, hh in labels:
            new_x = 1 - x
            new_labels.append([cls, new_x, y, ww, hh])

    elif aug_type == 'flip_vertical':
        new_image = np.flipud(image)
        for cls, x, y, ww, hh in labels:
            new_y = 1 - y
            new_labels.append([cls, x, new_y, ww, hh])

    elif aug_type == 'rotate_90_clockwise':
        new_image = rotate(image, -90, reshape=True)  # Negative for clockwise in scipy
        for cls, x, y, ww, hh in labels:
            new_x = 1 - y
            new_y = x
            new_w = hh
            new_h = ww
            new_labels.append([cls, new_x, new_y, new_w, new_h])

    elif aug_type == 'rotate_90_counter_clockwise':
        new_image = rotate(image, 90, reshape=True)
        for cls, x, y, ww, hh in labels:
            new_x = y
            new_y = 1 - x
            new_w = hh
            new_h = ww
            new_labels.append([cls, new_x, new_y, new_w, new_h])

    elif aug_type == 'rotate_180':
        new_image = rotate(image, 180, reshape=True)
        for cls, x, y, ww, hh in labels:
            new_x = 1 - x
            new_y = 1 - y
            new_labels.append([cls, new_x, new_y, ww, hh])

    else:
        raise ValueError("Unknown augmentation type")

    # Save new image
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    new_image_name = f"{base_name}_{output_suffix}{ext}"
    new_image_path = os.path.join(train_images_dir, new_image_name)
    plt.imsave(new_image_path, new_image)

    # Save new labels
    new_label_name = f"{base_name}_{output_suffix}.txt"
    new_label_path = os.path.join(train_labels_dir, new_label_name)
    save_labels(new_label_path, new_labels)


# Now apply augmentations to each remaining train file
augmentations = [
    ('flip_horizontal', 'flip_h'),
    ('flip_vertical', 'flip_v'),
    ('rotate_90_clockwise', 'rot_90_cw'),
    ('rotate_90_counter_clockwise', 'rot_90_ccw'),
    ('rotate_180', 'rot_180')
]

train_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  # Refresh list

for file in train_files:
    image_path = os.path.join(train_images_dir, file)
    label_file = os.path.splitext(file)[0] + '.txt'
    label_path = os.path.join(train_labels_dir, label_file)
    if not os.path.exists(label_path):
        print(f"Skipping {file}: no label file")
        continue

    for aug_type, suffix in augmentations:
        augment_image_and_labels(image_path, label_path, aug_type, suffix)

print("Data augmentation completed. Original train images are kept, augmented versions added with suffixes.")