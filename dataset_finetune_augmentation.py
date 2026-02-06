import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ============================================================
# Falcon raw label IDs  →  training IDs (0..9)
# ============================================================
RAW_TO_TRAIN = {
    100: 0,     # Sky
    200: 1,     # Landscape
    300: 2,     # Dry grass
    500: 3,     # Trees
    550: 4,     # Bushes
    600: 5,     # Rocks
    700: 6,     # Logs
    800: 7,     # Ground clutter
    7100: 8,    # Lush bushes
    10000: 9    # Background
}


class OffroadSegDataset(Dataset):
    def __init__(self, root, split="train"):
        """
        Args:
            root (str): Dataset root directory
            split (str): 'train' or 'val'
        """
        self.root = root
        self.split = split

        self.img_dir = os.path.join(root, split, "Color_Images")
        self.mask_dir = os.path.join(root, split, "Segmentation")

        self.fnames = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # ------------------ Load images ------------------
        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path)

        # ------------------ Augmentations (TRAIN ONLY) ------------------
        if self.split == "train":

            # Random horizontal flip
            if random.random() > 0.5:
                img_pil = TF.hflip(img_pil)
                mask_pil = TF.hflip(mask_pil)

            # Random brightness / contrast
            img_pil = TF.adjust_brightness(
                img_pil, 1.0 + random.uniform(-0.2, 0.2)
            )
            img_pil = TF.adjust_contrast(
                img_pil, 1.0 + random.uniform(-0.2, 0.2)
            )

        # ------------------ Convert to numpy ------------------
        img = np.array(img_pil, dtype=np.uint8)
        raw_mask = np.array(mask_pil, dtype=np.int32)

        # ------------------ Remap raw labels → train labels ------------------
        mask = np.zeros_like(raw_mask, dtype=np.int64)

        for raw_val, train_id in RAW_TO_TRAIN.items():
            mask[raw_mask == raw_val] = train_id

        # ------------------ To torch tensors ------------------
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return img, mask
