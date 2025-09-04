import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# ==== Dataset ====
class CULaneDataset(Dataset):
    def __init__(self, image_root, label_root, image_transform=None, label_transform=None):
        self.samples = []
        for root, _, files in os.walk(image_root):
            for file in files:
                if file.endswith(".jpg"):
                    rel_path = os.path.relpath(os.path.join(root, file), image_root)
                    image_path = os.path.join(image_root, rel_path)
                    label_path = os.path.join(label_root, rel_path).replace(".jpg", ".png")
                    if os.path.exists(label_path):
                        self.samples.append((image_path, label_path))
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = Image.open(lbl_path).convert('L')
        if self.image_transform: image = self.image_transform(image)
        if self.label_transform: label = self.label_transform(label)
        return image, label

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

