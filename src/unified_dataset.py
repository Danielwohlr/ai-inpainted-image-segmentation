import os
import numpy as np
from torch.utils.data import Dataset


class UnifiedDataset(Dataset):
    """
    Loads both manipulated and original images in a single dataset.
    We keep a label is_original[i] to know if it's an original or manipulated.
    """

    def __init__(
        self, manip_image_dir, manip_mask_dir, originals_dir=None, transform=None
    ):
        """
        Args:
          manip_image_dir: path to .npy feature files for manipulated images
          manip_mask_dir: path to .npy mask files for manipulated images
          originals_dir: path to .npy feature files for original images (zero masks)
          transform: albumentations transform
        """
        super().__init__()
        self.transform = transform

        # 1) Gather manipulated images
        self.manip_image_paths = sorted(
            [
                os.path.join(manip_image_dir, f)
                for f in os.listdir(manip_image_dir)
                if f.endswith(".npy")
            ]
        )
        self.manip_mask_paths = sorted(
            [
                os.path.join(manip_mask_dir, f)
                for f in os.listdir(manip_mask_dir)
                if f.endswith(".npy")
            ]
        )
        assert len(self.manip_image_paths) == len(
            self.manip_mask_paths
        ), "Mismatch between manipulated images and masks!"

        # We'll store them in a list of (image_path, mask_path, is_original=False)
        self.samples = []
        for img_path, msk_path in zip(self.manip_image_paths, self.manip_mask_paths):
            self.samples.append((img_path, msk_path, False))

        # 2) Gather originals if provided
        if originals_dir and os.path.isdir(originals_dir):
            original_paths = sorted(
                [
                    os.path.join(originals_dir, f)
                    for f in os.listdir(originals_dir)
                    if f.endswith(".npy")
                ]
            )
            for op in original_paths:
                # We'll store (image_path, None, True)
                # and we'll create an all-zero mask in __getitem__
                self.samples.append((op, None, True))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, is_original = self.samples[idx]

        # Load image features
        image = np.load(img_path)  # shape e.g. (256,256,7)

        # If it's an original, create a zero mask. Otherwise, load from manip_mask_dir
        if is_original:
            H, W, _ = image.shape
            mask = np.zeros((H, W, 1), dtype=np.float32)
        else:
            mask = np.load(mask_path)  # shape (256,256,1) typically

        # Albumentations transform
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # Ensure mask shape is (C,H,W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.permute(2, 0, 1)

            mask = mask.float()

        return image, mask

    def get_labels(self):
        """
        Returns a list of 0/1 labels indicating manipulated (0) or original (1)
        for each sample. Used for WeightedRandomSampler.
        """
        labels = []
        for _, _, is_original in self.samples:
            labels.append(1 if is_original else 0)
        return labels
