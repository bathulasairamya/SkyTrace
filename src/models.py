"""Model Architectures for Satellite Building Segmentation.

Includes:
1. DeepLabV3Plus: Exactly matches the weights in best_deeplab.pth (torchvision ResNet-50 backbone)
2. AuthorUNet: Matches best_unet.pth (5-stage DoubleConv U-Net with BatchNorm)
3. PaperOptimizedUNet: Structurally optimized 32-512 U-Net from Pasquali et al. (Remote Sens. 2019)
"""

import os
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet-50 backbone.
    Matches the exact architecture used to train checkpoints/best_deeplab.pth.
    """
    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        weights = "DEFAULT" if pretrained_backbone else None
        self.model = deeplabv3_resnet50(weights=weights)
        # Adapt output classifier head to 1 channel (building foreground logit)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        # Aux classifier also adapted if present
        if hasattr(self.model, "aux_classifier") and self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        out = self.model(x)
        return out["out"]


class DoubleConv(nn.Module):
    """Standard Double Convolution block with BatchNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AuthorUNet(nn.Module):
    """5-stage U-Net matching checkpoints/best_unet.pth."""
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        b = self.bottleneck(self.pool4(d4))

        u4 = self.conv4(torch.cat([self.up4(b), d4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))

        return self.out(u1)


class PaperOptimizedUNet(nn.Module):
    """Structurally optimized 32-512 U-Net from Pasquali et al. (Remote Sens. 2019).
    Key features:
    - Initial layer depth = 32 (saves GPU memory).
    - Deepest 1024-layer removed (prevents vanishing gradients).
    - No BatchNorm (better convergence on small batch sizes = 2).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.d1 = conv_block(in_channels, 32)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = conv_block(32, 64)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = conv_block(64, 128)
        self.p3 = nn.MaxPool2d(2)

        self.d4 = conv_block(128, 256)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck at 512 channels (1024 removed)
        self.b = conv_block(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c4 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c1 = conv_block(64, 32)

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))

        b = self.b(self.p4(d4))

        u4 = self.c4(torch.cat([self.up4(b), d4], dim=1))
        u3 = self.c3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.c2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.c1(torch.cat([self.up1(u2), d1], dim=1))

        return self.out(u1)


def build_model(model_name: str = "deeplabv3", checkpoint_path: str = None, device: str = "cpu"):
    """Factory function to build model and optionally load checkpoint with verification."""
    model_name = model_name.lower()
    if "deeplab" in model_name:
        model = DeepLabV3Plus()
    elif "author_unet" in model_name or "author" in model_name:
        model = AuthorUNet()
    elif "paper" in model_name or "unet" in model_name:
        model = PaperOptimizedUNet()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # If wrapped in custom container or top-level keys
        model.load_state_dict(state_dict, strict=False)
        print(f"[ModelFactory] Loaded checkpoint '{checkpoint_path}' into {type(model).__name__}")

    model.to(device)
    return model


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing models on device: {dev}")
    x = torch.randn(2, 3, 512, 512).to(dev)

    dl = DeepLabV3Plus().to(dev)
    print("DeepLabV3Plus output shape:", dl(x).shape)

    un = AuthorUNet().to(dev)
    print("AuthorUNet output shape:", un(x).shape)

    pn = PaperOptimizedUNet().to(dev)
    print("PaperOptimizedUNet output shape:", pn(x).shape)
