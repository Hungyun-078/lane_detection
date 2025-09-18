import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet  
from dataset import CULaneDataset 
from pathlib import Path

INPUT_H, INPUT_W = 288, 800

def save_visuals(imgs, masks, logits, out_dir: Path, start_idx=0, thresh=0.35):
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i in range(imgs.size(0)):
        if saved >= 35:
            break
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].imshow(imgs[i].cpu().permute(1, 2, 0))
        axs[0].set_title("Input"); axs[0].axis('off')
        axs[1].imshow(masks[i].cpu().squeeze(), cmap='gray')
        axs[1].set_title("label"); axs[1].axis('off')
        axs[2].imshow(torch.sigmoid(logits[i]).cpu().squeeze() > thresh, cmap='gray')
        axs[2].set_title("Prediction"); axs[2].axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f"sample_{start_idx + i}.png")
        plt.close()
        saved += 1

print("=== load model ===")

model_path = "/home/scill/unet_lane_detection/runs_culane_unet_182_30/best_f1_model.pth"
image_dir = "/home/scill/Downloads/CULane/driver_182_30frame/driver_182_30frame"
label_dir = "/home/scill/Downloads/CULane/laneseg_label_w16/laneseg_label_w16/driver_182_30frame"

image_transform = transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
	transforms.ToTensor(),
])
label_transform = transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0).float())
])

dataset = CULaneDataset(image_dir, label_dir, image_transform, label_transform)
test_dataset = Subset(dataset, range(9000, min(10000, len(dataset)))) 
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)


device = torch.device("cuda")
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("===== start =====")

# === 評估每一個 batch 後做平均 ===
TP = 0
FP = 0
FN = 0


with torch.no_grad():
        sample = 0
        for imgs, masks in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.35).float()

            # 拉平成 1D（像素級）
            y_true = masks.view(-1)
            y_pred = preds.view(-1)

            # 累計 TP/FP/FN
            TP += (y_pred * y_true).sum().item()
            FP += (y_pred * (1 - y_true)).sum().item()
            FN += (((1 - y_pred) * y_true)).sum().item()
            sample += imgs.size(0)
            print(f"sample:{sample}",end = "\r")

test_dir = Path("/home/scill/unet_lane_detection/runs_culane_unet_182_30/test")
save = 0
with torch.no_grad():
        for imgs, masks in test_loader:
            logits = model(imgs.to(device))
            save_visuals(imgs, masks, logits.cpu(), test_dir, start_idx=save, thresh=0.35)
            save += imgs.size(0)
            if save >= 35:
               break

eps = 1e-9
precision = TP / (TP + FP + eps)
recall    = TP / (TP + FN + eps)
f1        = 2 * precision * recall / (precision + recall + eps)

print("\n======= result=======")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" F1 Score:  {f1:.4f}")
