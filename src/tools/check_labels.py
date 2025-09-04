import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np  # For potential array handling, though not strictly needed


def visualize_detection():
    """
    Visualize bounding boxes and class IDs on an image using normalized YOLO-style labels.

    Args:
    - image_path (str): Path to the image file (e.g., 'image.jpg').
    - label_path (str): Path to the label file (e.g., 'label.txt') in YOLO format:
                        Each line: class_id x_center y_center width height (normalized 0-1).
    - output_path (str, optional): If provided, save the visualized image to this path instead of showing it.

    Example usage:
    visualize_detection('path/to/image.jpg', 'path/to/label.txt')
    """
    # Load the image
    image_path = r"E:\fish\comp\origin\train\images\2022-06-10-13-58-32-64149_2048-1536_aug0.png"
    label_path = r"E:\fish\comp\origin\train\labels\2022-06-10-13-58-32-64149_2048-1536_aug0.txt"
    img = plt.imread(image_path)
    if img.dtype == np.float32 or img.dtype == np.float64:  # Ensure image is in 0-255 if needed
        img = (img * 255).astype(np.uint8)

    # Get image dimensions
    height, width = img.shape[:2]  # Handles both RGB and grayscale

    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    # Read labels and draw boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip invalid lines
            class_id = parts[0]
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            bbox_width = float(parts[3]) * width
            bbox_height = float(parts[4]) * height

            # Convert to top-left corner coordinates
            x_top_left = x_center - bbox_width / 2
            y_top_left = y_center - bbox_height / 2

            # Draw rectangle
            rect = patches.Rectangle(
                (x_top_left, y_top_left),
                bbox_width,
                bbox_height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add class ID text (above the box)
            ax.text(
                x_top_left,
                y_top_left - 5,  # Slightly above the box
                f'ID: {class_id}',
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=12,
                color='white'
            )

    # Remove axes for cleaner display
    ax.axis('off')


    plt.show()
visualize_detection()
# Example: If you have multiple images and labels, you can loop like this:
# import os
# image_dir = 'path/to/images'
# label_dir = 'path/to/labels'
# for img_file in os.listdir(image_dir):
#     if img_file.endswith('.jpg'):
#         img_path = os.path.join(image_dir, img_file)
#         label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
#         if os.path.exists(label_path):
#             visualize_detection(img_path, label_path)