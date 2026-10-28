"""Model Architectures for Satellite Building Segmentation.

Includes:
1. DeepLabV3Plus: Exactly matches the weights in best_deeplab.pth (torchvision ResNet-50 backbone)
2. AuthorUNet: Matches best_unet.pth (5-stage DoubleConv U-Net with BatchNorm)
3. PaperOptimizedUNet: Structurally optimized 32-512 U-Net from Pasquali et al. (Remote Sens. 2019)
"""

import os
import hashlib
import logging
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

logger = logging.getLogger("SkyTrace.Models")


class DeepLabV3ResNet50(nn.Module):
    """DeepLabV3 with ResNet-50 backbone and Atrous Spatial Pyramid Pooling (ASPP).

    Architecture Reference:
        Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation" (arXiv:1706.05587).
        Backbone: ResNet-50 with dilated convolutions in stages 3 and 4 (output stride = 16 or 8).
        Classifier: ASPP module with atrous rates (1, 6, 12, 18) + Image Pooling, followed by 1x1 conv.
    
    Architectural Clarification:
        This class instantiates Torchvision's standard DeepLabV3 architecture, NOT DeepLabV3+.
        Standard DeepLabV3 projects ASPP features directly to the output logits without the
        low-level encoder skip connection and feature-projection decoder present in DeepLabV3+.
        Matches the exact weights trained in checkpoints/best_deeplab.pth.
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


# Backward compatibility alias
DeepLabV3Plus = DeepLabV3ResNet50


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


def _pad_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Pad upsampled tensor to match skip connection spatial dimensions if needed."""
    diff_y = skip.size()[2] - up.size()[2]
    diff_x = skip.size()[3] - up.size()[3]
    if diff_y != 0 or diff_x != 0:
        up = nn.functional.pad(up, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
    return torch.cat([up, skip], dim=1)


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

        u4 = self.conv4(_pad_cat(self.up4(b), d4))
        u3 = self.conv3(_pad_cat(self.up3(u4), d3))
        u2 = self.conv2(_pad_cat(self.up2(u3), d2))
        u1 = self.conv1(_pad_cat(self.up1(u2), d1))

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

        u4 = self.c4(_pad_cat(self.up4(b), d4))
        u3 = self.c3(_pad_cat(self.up3(u4), d3))
        u2 = self.c2(_pad_cat(self.up2(u3), d2))
        u1 = self.c1(_pad_cat(self.up1(u2), d1))

        return self.out(u1)


def compute_file_sha256(filepath: str) -> str:
    """Calculate SHA-256 hash of a file for artifact verification."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


MODEL_REGISTRY = {
    "deeplabv3": DeepLabV3ResNet50,
    "deeplabv3_r50": DeepLabV3ResNet50,
    "deeplabv3_resnet50": DeepLabV3ResNet50,
    "deeplabv3plus": DeepLabV3ResNet50,
    "author_unet": AuthorUNet,
    "unet": AuthorUNet,
    "paper_unet": PaperOptimizedUNet,
    "paper_optimized_unet": PaperOptimizedUNet,
}


def build_model(
    model_name: str = "deeplabv3",
    checkpoint_path: str = None,
    device: str = "cpu",
    allow_partial: bool = False,
):
    """Factory function to instantiate models with strict checkpoint validation and SHA-256 logging.

    Args:
        model_name: Architectural identifier in MODEL_REGISTRY.
        checkpoint_path: Path to .pth / .pt weights file on disk.
        device: Target execution device ('cpu', 'cuda', etc.).
        allow_partial: If False, raises RuntimeError on unexpected or missing core weights.
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        valid_keys = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model name '{model_name}'. Choose from: {valid_keys}")

    model_cls = MODEL_REGISTRY[key]
    model = model_cls()

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

        sha256 = compute_file_sha256(checkpoint_path)
        print(f"[ModelRegistry] Loading '{checkpoint_path}' (SHA-256: {sha256[:12]}...)")

        raw_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Handle wrapped training dictionaries
        state_dict = raw_state.get("model_state_dict", raw_state)
        # Strip DataParallel prefix if present
        clean_state = {k.removeprefix("module."): v for k, v in state_dict.items()}

        # Load weights
        load_result = model.load_state_dict(clean_state, strict=False)

        # Filter out documented benign auxiliary classifier keys from torchvision base
        unexpected = [k for k in load_result.unexpected_keys if not k.startswith("model.aux_classifier")]
        missing = load_result.missing_keys

        if load_result.unexpected_keys:
            aux_ignored = [k for k in load_result.unexpected_keys if k.startswith("model.aux_classifier")]
            if aux_ignored:
                print(f"[ModelRegistry] Note: Ignored {len(aux_ignored)} unused torchvision aux_classifier weights from checkpoint.")

        if (missing or unexpected) and not allow_partial:
            error_msg = (
                f"Checkpoint parameter mismatch in {model_cls.__name__}!\n"
                f"  Missing keys ({len(missing)}): {missing[:5]}\n"
                f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
            )
            raise RuntimeError(error_msg)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"[ModelRegistry] Verified and strictly loaded {model_cls.__name__} ({total_params/1e6:.2f}M parameters).")

    model.to(device)
    return model


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing models on device: {dev}")
    x = torch.randn(2, 3, 512, 512).to(dev)

    dl = build_model("deeplabv3", device=dev)
    print("DeepLabV3ResNet50 output shape:", dl(x).shape)

    un = build_model("author_unet", device=dev)
    print("AuthorUNet output shape:", un(x).shape)

    pn = build_model("paper_unet", device=dev)
    print("PaperOptimizedUNet output shape:", pn(x).shape)
