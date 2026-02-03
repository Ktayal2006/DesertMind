import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import OffroadSegDataset
from torch.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True


def main():
    # ================= CONFIG =================
    ROOT = r"C:\Users\DELL\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset"
    NUM_CLASSES = 10
    BATCH_SIZE = 3
    LR = 1e-4

    EPOCHS = 20          # ⬅ allow training, early stopping will cut it
    PATIENCE = 3         # ⬅ stop if no improvement for 3 epochs

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "best_deeplab.pth"

    print("Using device:", DEVICE)

    # ================= mIoU =================
    @torch.no_grad()
    def compute_miou(pred, target, num_classes=10):
        ious = []
        for cls in range(num_classes):
            pred_i = (pred == cls)
            target_i = (target == cls)
            intersection = (pred_i & target_i).sum().item()
            union = (pred_i | target_i).sum().item()
            if union == 0:
                continue
            ious.append(intersection / union)
        return float(np.mean(ious)) if ious else 0.0

    # ================= DATA =================
    train_ds = OffroadSegDataset(ROOT, split="train")
    val_ds   = OffroadSegDataset(ROOT, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    print("Train size:", len(train_ds), "Val size:", len(val_ds))

    # ================= MODEL =================
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler("cuda")

    # ================= TRAIN =================
    best_miou = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                out = model(imgs)["out"]
                loss = criterion(out, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_losses = []
        val_ious = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                with autocast(device_type="cuda"):
                    out = model(imgs)["out"]
                    loss = criterion(out, masks)

                val_losses.append(loss.item())
                pred = torch.argmax(out, dim=1)
                miou = compute_miou(pred.cpu(), masks.cpu(), NUM_CLASSES)
                val_ious.append(miou)

        val_loss = float(np.mean(val_losses))
        val_miou = float(np.mean(val_ious))

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val mIoU {val_miou:.4f}"
        )

        # ================= EARLY STOPPING =================
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_without_improvement = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print("   Saved best model:", best_miou)
        else:
            epochs_without_improvement += 1
            print(f"   No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training done. Best validation mIoU:", best_miou)


if __name__ == "__main__":
    main()
