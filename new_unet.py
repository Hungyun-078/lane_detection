import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import matplotlib.pyplot as plt

from unet import UNet
from dataset import CULaneDataset

# -----------------------------
# Config
# -----------------------------
IMAGE_DIR = "/home/scill/Downloads/CULane/driver_182_30frame/driver_182_30frame"
LABEL_DIR = "/home/scill/Downloads/CULane/laneseg_label_w16/laneseg_label_w16/driver_182_30frame"
OUT_DIR = Path("runs_culane_unet_182_30")
CKPT_PATH = OUT_DIR / "checkpoint.pth"
TRAIN_SAMPLE = 8000
VAL_SAMPLE = 1000
EPOCHS = 50
BATCH_SIZE = 4
BASE_LR = 1e-3
INPUT_H, INPUT_W = 288, 800
THRESH = 0.35
SEED = 1337

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(logits.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, bce_w: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()
        self.bce_w = bce_w
    def to(self, device):
        self.bce.pos_weight = self.bce.pos_weight.to(device)
        return super().to(device)
    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + (1 - self.bce_w) * self.dice(logits, targets)

def save_visuals(imgs, masks, preds, out_dir: Path, start_idx=0, max_to_save=40, thresh=0.35):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i in range(imgs.size(0)):
        if saved >= max_to_save:
            break
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].imshow(imgs[i].cpu().permute(1, 2, 0))
        axs[0].set_title("Input"); axs[0].axis('off')
        axs[1].imshow(masks[i].cpu().squeeze(), cmap='gray')
        axs[1].set_title("GT"); axs[1].axis('off')
        axs[2].imshow((torch.sigmoid(preds[i:i+1]).cpu().squeeze()>thresh), cmap='gray')
        axs[2].set_title("Pred"); axs[2].axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f"sample_{start_idx + i}.png")
        plt.close()
        saved += 1

# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
    ])
    label_transform = transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0).float())
    ])

    # Dataset
    ds_full = CULaneDataset(IMAGE_DIR, LABEL_DIR, image_transform, label_transform) 
    ds_full = Subset(ds_full, range(min(9000, len(ds_full))))
 
    # Split train/val
    total_len = len(ds_full)
    val_len = min(VAL_SAMPLE, total_len)
    train_len = min(TRAIN_SAMPLE, total_len-val_len)
    train_ds, val_ds = random_split(ds_full, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    model = UNet().to(device)

    # Loss & optimizer
    loss_fn = BCEDiceLoss(pos_weight=1.0, bce_w=0.6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Resume checkpoint
    start_epoch = 0
    best_f1 = -1.0
    loss_history = []
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        loss_history = ckpt.get('loss_history', [])
        start_epoch = ckpt.get('epoch', -1) + 1
        best_f1 = ckpt.get('best_f1', -1.0)
        print(f"✅ Resumed from epoch {start_epoch}, best F1={best_f1:.4f}")

    # ---- Training Loop ----
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        batches = 0
        samples_seen = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if masks.max() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batches += 1
            samples_seen += imgs.size(0)
            print(f"Epoch {epoch+1}/{EPOCHS}, Sample {samples_seen}", end='\r')

        avg_train_loss = running_loss / max(1, batches)
        loss_history.append(avg_train_loss)

        # ---- Validation ----
        model.eval()
        val_loss_acc = 0.0
        val_batches = 0
        TP = FP = FN = 0.0
        eps = 1e-6

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    logits = model(imgs)
                    val_loss = loss_fn(logits, masks)

                val_loss_acc += val_loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits)
                preds = (probs > THRESH).float()

                # 拉平成 1D
                y_true = masks.view(-1)
                y_pred = preds.view(-1)

                TP += (y_pred * y_true).sum().item()
                FP += (y_pred * (1 - y_true)).sum().item()
                FN += ((1 - y_pred) * y_true).sum().item()

        avg_val_loss = val_loss_acc / max(1, val_batches)
        P_micro = TP / (TP + FP + eps)
        R_micro = TP / (TP + FN + eps)
        F1_micro = 2 * P_micro * R_micro / (P_micro + R_micro + eps)
        mean_iou = TP / (TP + FP + FN + eps)

        scheduler.step(avg_val_loss)
        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{EPOCHS} | train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | F1={F1_micro:.4f} "
            f"P={P_micro:.4f} R={R_micro:.4f} IoU={mean_iou:.4f} | {dt:.1f}s"
        )

        # ---- Save checkpoint ----
        is_best = F1_micro > best_f1
        best_f1 = max(best_f1, F1_micro)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss_history': loss_history,
            'best_f1': best_f1,
            'threshold': THRESH,
        }, CKPT_PATH)
        if is_best:
            torch.save(model.state_dict(), OUT_DIR / "best_f1_model.pth")

        # ---- Save sample predictions ----
        preds_dir = OUT_DIR / f"predictions_epoch_{epoch+1}"
        saved = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                logits = model(imgs.to(device))
                save_visuals(imgs, masks, logits.cpu(), preds_dir, start_idx=saved,
                             max_to_save=35, thresh=THRESH)
                saved += imgs.size(0)
                if saved >= 35:
                    break

    # ---- Final artifacts ----
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.grid(True); plt.legend()
    plt.savefig(OUT_DIR / 'loss_curve.png')
    torch.save(model.state_dict(), OUT_DIR / "model_final.pth")
    print(f"\n✅ Done. Best F1={best_f1:.4f}. Artifacts at {OUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
