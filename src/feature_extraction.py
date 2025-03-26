"""
Used to normalize RGB values and extract Canny, Sobel, Fourier, and Laplacian features from images.
These features are saved as numpy arrays in the processed/train/images directory.
Also transforms masks to a single channel.

Fun extract_features is also used during inference on-the-fly.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# Set True if creating features for test set (no masks)
ORIGINAL = True
# Paths
if not ORIGINAL:
    image_dir = "dataset/train/train/images"
    mask_dir = "dataset/train/train/masks"
    out_img_dir = "processed/train/images"
    out_mask_dir = "processed/train/masks"

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
else:
    image_dir = "dataset/train/train/originals"
    out_img_dir = "processed/train/originals"
    os.makedirs(out_img_dir, exist_ok=True)


def normalize_channel(chan):
    chan -= chan.min()
    chan /= chan.max() + 1e-8
    return chan


def extract_features(img):
    """
    Transforms images: normalizes RGB with ImageNet stats, and scales other channels to [0,1].
    """
    # Convert to float and normalize RGB to [0,1]
    rgb = img[..., :3].astype(np.float32) / 255.0

    # Normalize RGB channels using ImageNet stats
    rgb[..., 0] = (rgb[..., 0] - 0.485) / 0.229  # R
    rgb[..., 1] = (rgb[..., 1] - 0.456) / 0.224  # G
    rgb[..., 2] = (rgb[..., 2] - 0.406) / 0.225  # B

    # Grayscale version for edge detectors
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_norm = gray / 255.0

    # Canny (already in 0–255, so divide to get 0–1)
    canny = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0

    # Sobel magnitude → normalize to [0,1]
    sobelx = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = normalize_channel(sobel_mag)

    # Fourier magnitude (log-scaled) → already normalized
    f = np.fft.fft2(gray_norm)
    fshift = np.fft.fftshift(f)
    ft_mag = np.log1p(np.abs(fshift))
    ft_mag = ft_mag / ft_mag.max()

    # Laplacian → normalize to [0,1]
    laplacian = cv2.Laplacian(gray_norm, cv2.CV_64F)
    laplacian = normalize_channel(laplacian)

    # Stack all 7 channels
    stacked = np.dstack(
        [
            rgb,
            canny[..., None],
            sobel_mag[..., None],
            ft_mag[..., None],
            laplacian[..., None],
        ]
    )

    return stacked.astype(np.float32)


if __name__ == "__main__":

    # Do this only to generate features for training
    image_files = sorted(os.listdir(image_dir))

    for fname in tqdm(image_files):
        # Load image
        img = cv2.imread(os.path.join(image_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = extract_features(img)
        # Check for NaNs in features
        if np.isnan(features).any():
            print(f"NaNs found in image features: {fname}")
            continue
        if not ORIGINAL:
            # Load mask
            mask = cv2.imread(os.path.join(mask_dir, fname))
            binary_mask = (np.any(mask > 0, axis=-1)).astype(np.uint8)[..., None]

            np.save(
                os.path.join(out_mask_dir, fname.replace(".png", ".npy")), binary_mask
            )
        # Save
        np.save(os.path.join(out_img_dir, fname.replace(".png", ".npy")), features)
