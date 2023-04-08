import os

import cv2
import numpy as np
from tqdm import tqdm


def compute_mean_std_train_only(datapath):
    """
    Compute the mean and standard deviation of all images in the training dataset only.

    Args:
        datapath (str): Path to the dataset directory.

    Returns:
        tuple: mean and std of the training dataset across all channels.
    """
    train_img_dir = os.path.join(datapath, 'train', 'IMG', 'train')  # Corrected path to training images
    img_paths = [os.path.join(train_img_dir, fname) for fname in os.listdir(train_img_dir) if fname.endswith('.png')]

    pixel_num = 0
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)

    for img_path in tqdm(img_paths, desc='Computing mean and std of training data'):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
        pixel_num += (img.size / 3)
        channel_sum += img.sum(axis=(0, 1))
        channel_sum_squared += (img ** 2).sum(axis=(0, 1))

    mean = channel_sum / pixel_num
    std = np.sqrt(channel_sum_squared / pixel_num - mean ** 2)

    return mean, std


datapath = '/Users/bob/PycharmProjects/UAV-2023/dataset'
mean, std = compute_mean_std_train_only(datapath)
print("Mean of Training Data:", mean)
print("Std of Training Data:", std)
