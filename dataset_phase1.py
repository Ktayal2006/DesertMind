import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Global mapping: Falcon raw labels -> 0..9
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

        # RGB image
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Raw mask
        raw_mask = np.array(Image.open(mask_path), dtype=np.int32)

        # Remap labels
        mask = np.zeros_like(raw_mask, dtype=np.int64)
        for raw_val, train_id in RAW_TO_TRAIN.items():
            mask[raw_mask == raw_val] = train_id

        mask = torch.from_numpy(mask)
        return img, mask
