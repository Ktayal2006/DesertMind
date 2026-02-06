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
    def __init__(self, root, split="train", crop_size=512):
        self.split = split
        self.crop_size = crop_size

        self.img_dir = os.path.join(root, split, "Color_Images")
        self.mask_dir = os.path.join(root, split, "Segmentation")

        self.fnames = sorted(
            [f for f in os.listdir(self.img_dir) if f.endswith(".png")]
        )

    def __len__(self):
        return len(self.fnames)

    def random_crop(self, img, mask):
        H, W = mask.shape
        cs = self.crop_size

        if H < cs or W < cs:
            return img, mask  # safety

        y = random.randint(0, H - cs)
        x = random.randint(0, W - cs)

        img = img[:, y:y + cs, x:x + cs]
        mask = mask[y:y + cs, x:x + cs]
        return img, mask

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # ---- load image ----
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # ---- load mask ----
        raw_mask = np.array(Image.open(mask_path), dtype=np.int32)
        mask = np.zeros_like(raw_mask, dtype=np.int64)

        for raw_val, train_id in RAW_TO_TRAIN.items():
            mask[raw_mask == raw_val] = train_id

        mask = torch.from_numpy(mask)

        # ---- training-time augmentations ----
        if self.split == "train":
            img, mask = self.random_crop(img, mask)

            # horizontal flip
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[1])

        return img, mask
