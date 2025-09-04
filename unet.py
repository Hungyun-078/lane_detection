import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==== UNet Model ====
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.double_conv(3, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.double_conv(16, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.double_conv(64, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = self.double_conv(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(32, 16)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final_conv(d1)
    

