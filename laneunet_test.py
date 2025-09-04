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
model_path = "/home/scill/unet_lane_detection/model_final_10000.pth"

# === 設定資料與模型 ===
image_dir = "/home/scill/Downloads/CULane/driver_161_90frame/driver_161_90frame"
label_dir = "/home/scill/Downloads/CULane/laneseg_label_w16/laneseg_label_w16/driver_161_90frame"

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
f1_scores = []
precisions = []
recalls = []

with torch.no_grad():
    sample=0
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = torch.sigmoid(model(imgs))
        preds = (outputs > 0.6).float()

        # 轉為 numpy 形式
        y_true = masks.cpu().numpy().reshape(-1)
        y_pred = preds.cpu().numpy().reshape(-1)

        # 濾掉全 0 的 label（無標記）
        if y_true.max() == 0:
            continue

        # 計算每個 batch 的指標
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        sample += imgs.size(0)
        print(f"Samples: {sample}",end="\r")

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

# === 平均結果 ===
print("\n===== 測試集平均結果 (基於 batch) =====")
print(f"📊 Precision: {np.mean(precisions):.4f}")
print(f"📊 Recall:    {np.mean(recalls):.4f}")
print(f"📊 F1 Score:  {np.mean(f1_scores):.4f}")
