import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet  
from dataset import CULaneDataset 

INPUT_H, INPUT_W = 288, 800

print("=== 測試 CULane 資料集的 UNet 模型 ===")


# ==== 路徑與設定 ====
model_path = "/home/scill/unet_lane_detection/runs_culane_unet/model_final.pth"

# === 設定資料與模型 ===
image_dir = "/home/scill/Downloads/CULane/driver_161_90frame/driver_161_90frame"
label_dir = "/home/scill/Downloads/CULane/laneseg_label_w16/laneseg_label_w16/driver_161_90frame"

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
test_dataset = Subset(dataset, range(10000, min(11000, len(dataset)))) 
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

# === 載入模型 ===
device = torch.device("cuda" )
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("模型載入完成，開始測試...")

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

            # 累計 TP/FP/FN（注意：不需要 TN 計算 F1）
            TP += (y_pred * y_true).sum().item()
            FP += (y_pred * (1 - y_true)).sum().item()
            FN += (((1 - y_pred) * y_true)).sum().item()
            sample += img.size(0)
            print(f"sample:{sample}",end = "\r")

precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)

print("\n===== 測試集平均結果 (基於 batch) =====")
print(f" Precision: {np.mean(precision):.4f}")
print(f" Recall:    {np.mean(recall):.4f}")
print(f" F1 Score:  {np.mean(f1):.4f}")
