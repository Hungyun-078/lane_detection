import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from unet import UNet
from dataset import CULaneDataset
from torchvision import transforms

# ==== Paths ====
image_dir = "/home/scill/Downloads/CULane/driver_161_90frame/driver_161_90frame"
label_dir = "/home/scill/Downloads/CULane/laneseg_label_w16/laneseg_label_w16/driver_161_90frame"
checkpoint_path = "checkpoint.pth"

# ==== Transform ====
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
label_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0).float())
])

# ==== Dataset ====
dataset_full = CULaneDataset(image_dir, label_dir, image_transform, label_transform)
limited_dataset = Subset(dataset_full, range(min(10000, len(dataset_full))))
dataloader = DataLoader(limited_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# ==== Model Setup ====
device = torch.device("cuda")
model = UNet().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

# ==== Load from checkpoint ====
start_epoch = 0
loss_history = []

if os.path.exists(checkpoint_path):
    print("🔄 載入 checkpoint 中...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    loss_history = checkpoint['loss_history']
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ 成功從第 {start_epoch} 個 epoch 繼續訓練")

# ==== Training ====
try:
    print("🚀 開始訓練 UNet 模型...")
    for epoch in range(start_epoch, 50):
        model.train()
        total_loss = 0
        sample = 0
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)

            if masks.max() == 0:
                continue

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)

            if torch.isnan(loss):
                print("🛑 NaN loss! Skipping.")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            sample += imgs.size(0)
            print(f"Epoch {epoch+1}, Sample {sample}", end="\r")

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"\n📉 Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # ==== Save checkpoint ====
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss_history': loss_history
        }, checkpoint_path)
        print(f"💾 Checkpoint 已儲存：epoch {epoch+1}")

        # ==== 儲存預測圖像（最多 50 張） ====
        model.eval()
        os.makedirs(f"predictions_10000/epoch{epoch+1}", exist_ok=True)
        save_num = 0
        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(dataloader):
                imgs, masks = imgs.to(device), masks.to(device)
                preds = torch.sigmoid(model(imgs)).cpu()
                imgs = imgs.cpu()
                masks = masks.cpu()

                for i in range(len(imgs)):
                    if save_num >= 50: break
                    save_num += 1
                    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
                    axs[0].imshow(imgs[i].permute(1, 2, 0))
                    axs[0].set_title("Input"); axs[0].axis('off')
                    axs[1].imshow(masks[i].squeeze(), cmap='gray')
                    axs[1].set_title("Ground Truth"); axs[1].axis('off')
                    axs[2].imshow(preds[i].squeeze() > 0.3, cmap='gray')
                    axs[2].set_title("Prediction"); axs[2].axis('off')
                    plt.tight_layout()
                    img_id = batch_idx * dataloader.batch_size + i
                    plt.savefig(f"predictions_10000/epoch{epoch+1}/sample{img_id}.png")
                    plt.close()

except KeyboardInterrupt:
    print("\n🛑 訓練被中斷，儲存中...")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss_history': loss_history
    }, checkpoint_path)
    print("💾 中斷時進度已儲存為 checkpoint.pth")

# ==== Save Final Loss Curve ====
plt.figure()
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_10000.png')
print("📈 Loss curve saved to loss_curve_10000.png")

# ==== Save Final Model ====
torch.save(model.state_dict(), "model_final_10000.pth")
print("✅ 最終模型已儲存為 model_final_10000.pth")
