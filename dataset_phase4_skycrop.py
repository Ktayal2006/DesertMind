import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

RAW_TO_TRAIN = {
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    600: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

class OffroadSegDataset(Dataset):
    def __init__(self, root, split="train"):
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

    def crop_top(self, img, mask):
        H, W = img.shape[:2]
        crop_h = int(0.2 * H)

        img = img[crop_h:, :, :]
        mask = mask[crop_h:, :]

        return img, mask

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path)

        # --- Convert to numpy ---
        img = np.array(img_pil, dtype=np.uint8)
        raw_mask = np.array(mask_pil, dtype=np.int32)

        # --- 🔥 NEW: SKY CROP ---
        img, raw_mask = self.crop_top(img, raw_mask)

        # --- Augmentations (same as Phase 2) ---
        if self.split == "train":
            if random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
                raw_mask = np.flip(raw_mask, axis=1).copy()

            img_pil = Image.fromarray(img)
            img_pil = TF.adjust_brightness(img_pil, 1.0 + random.uniform(-0.2, 0.2))
            img_pil = TF.adjust_contrast(img_pil, 1.0 + random.uniform(-0.2, 0.2))
            img = np.array(img_pil)

        # --- Remap labels ---
        mask = np.zeros_like(raw_mask, dtype=np.int64)
        for raw_val, train_id in RAW_TO_TRAIN.items():
            mask[raw_mask == raw_val] = train_id

        # --- To torch ---
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return img, mask
