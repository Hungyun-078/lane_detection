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
        preds = (probs > THRESH).float()

		#變成1D
        y_true = masks.view(-1).cpu().numpy().astype(np.uint8)
        y_pred = preds.view(-1).cpu().numpy().astype(np.uint8)

		if y_true.max() == 0:
            sample += imgs.size(0)
            print(f"Samples: {sample}", end="\r")
            continue

        # 計算每個 batch 的指標
		TP += np.sum((y_pred == 1) & (y_true == 1))
		FP += np.sum((y_pred == 1) & (y_true == 0))
        FN += np.sum((y_pred == 0) & (y_true == 1))

    
eps = 1e-9
precision_micro = TP / (TP + FP + eps)
recall_micro    = TP / (TP + FN + eps)
f1_micro        = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + eps)

print("===== 測試集整體結果（micro / 累計）=====")
print(f"Precision: {precision_micro:.4f}")
print(f"Recall:    {recall_micro:.4f}")
print(f"F1 Score:  {f1_micro:.4f}")
