from unet import UNet
import torch
import torch.nn as nn
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = UNet(n_classes=2, n_channels=1).to(device)

summary(model, (1, 572, 572))