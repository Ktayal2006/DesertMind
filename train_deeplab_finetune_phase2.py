import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from dataset import OffroadSegDataset
from torch.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True


@torch.no_grad()
def compute_miou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_i = (pred == cls)
        target_i = (target == cls)
        intersection = (pred_i & target_i).sum().item()
        union = (pred_i | target_i).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def compute_per_class_iou(pred, target, num_classes):
    ious = {}
    for cls in range(num_classes):
        pred_i = (pred == cls)
        target_i = (target == cls)

        intersection = (pred_i & target_i).sum().item()
        union = (pred_i | target_i).sum().item()

        if union == 0:
            ious[cls] = None
        else:
            ious[cls] = intersection / union
    return ious

@torch.no_grad()
def pixel_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def main():
    # ================= CONFIG =================
    ROOT = r"C:\Users\DELL\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset"
    NUM_CLASSES = 10
    BATCH_SIZE = 2
    LR = 1e-5
    EPOCHS = 6
    PATIENCE = 2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "best_deeplab.pth"

    print("Using device:", DEVICE)

    # ================= DATA =================
    train_ds = OffroadSegDataset(ROOT, split="train")
    val_ds   = OffroadSegDataset(ROOT, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
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
    model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.to(DEVICE)
    # ===== LOAD BEST CHECKPOINT (FOR FINE-TUNING) =====
    model.load_state_dict(torch.load(SAVE_PATH))
    print("Loaded best checkpoint for fine-tuning")

    # ===== UNFREEZE BACKBONE =====
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("Backbone unfrozen(gentle fine-tuning)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr=LR
    )

    scaler = GradScaler("cuda")

    best_miou = -1.0
    epochs_without_improvement = 0

    # ================= TRAIN =================
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
        per_class_ious = []   # RESET EACH EPOCH
        pixel_accs = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                with autocast(device_type="cuda"):
                    out = model(imgs)["out"]
                    loss = criterion(out, masks)

                pred = torch.argmax(out, dim=1)
                pixel_accs.append(pixel_accuracy(pred, masks))

                val_losses.append(loss.item())
                val_ious.append(compute_miou(pred.cpu(), masks.cpu(), NUM_CLASSES))
                per_class_ious.append(
                    compute_per_class_iou(pred.cpu(), masks.cpu(), NUM_CLASSES)
                )

        val_loss = float(np.mean(val_losses))
        val_miou = float(np.mean(val_ious))
        val_pixel_acc = float(np.mean(pixel_accs))

        print(
            f"\nEpoch {epoch}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val mIoU {val_miou:.4f} | "
            f"pixel acc {val_pixel_acc:.4f}"
        )

        # ================= PER-CLASS SUMMARY =================
        print("Per-class IoU:")
        for cls in range(NUM_CLASSES):
            values = [x[cls] for x in per_class_ious if x[cls] is not None]
            if values:
                print(f"  Class {cls}: {sum(values)/len(values):.4f}")
            else:
                print(f"  Class {cls}: N/A")

        # ================= EARLY STOPPING =================
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_without_improvement = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print("Saved best model:", best_miou)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training done. Best validation mIoU:", best_miou)


if __name__ == "__main__":
    main()
