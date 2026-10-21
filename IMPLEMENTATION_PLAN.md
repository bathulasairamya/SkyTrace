# Satellite Building Detection: End-to-End Implementation Plan

**Project:** Building Detection from High-Resolution Satellite Images  
**Pretrained Checkpoints:** [Hugging Face: Abhatta7/building-footprint-segmentation](https://huggingface.co/Abhatta7/building-footprint-segmentation)  
**Reference Paper:** *Building Footprint Extraction from Multispectral, Spaceborne Earth Observation Datasets Using a Structurally Optimized U-Net Convolutional Neural Network* (Pasquali, Iannelli, Dell'Acqua, MDPI Remote Sensing 2019, 11, 2803)  
**Dataset:** SpaceNet 2 Paris Buildings (`AOI_3_Paris`)

---

## 1. Project Overview & Architectural Alignment

This document outlines the step-by-step implementation for building footprint segmentation from high-resolution satellite imagery. It bridges the architectural specifications from Pasquali et al. (2019) with the pretrained checkpoints and source repository from Aryak Bhattacharya (DePaul University, 2026).

```
+-------------------------------------------------------------------------------------------------+
|                                        PROJECT PIPELINE                                         |
|                                                                                                 |
|   +-----------------------+     +-----------------------+     +-----------------------------+   |
|   |  SpaceNet-2 Paris     | --> | Deterministic Split   | --> | Untouched Test Partition    |   |
|   |  (PS-RGB + GeoJSON)   |     | (random_state=42)     |     | (Strict No-Leakage Audit)   |   |
|   +-----------------------+     +-----------------------+     +-----------------------------+   |
|                                                                             |                   |
|   +-----------------------+                                                 v                   |
|   | Pretrained Checkpoint | -------------------------------------> [ Baseline Inference ]      |
|   | (torchvision DLv3+)   |                                                 |                   |
|   +-----------------------+                                                 v                   |
|               |                                                   [ Metrics: IoU/Dice/P/R ]     |
|               v                                                             ^                   |
|   [ Fine-Tuning Pipeline ]                                                  |                   |
|   (BCE+Dice, 384-512 patch) ------------------------------------------------+                   |
|               |                                                                                 |
|               v                                                                                 |
|   +-----------------------+     +-----------------------------------------------------------+   |
|   | Best Model Checkpoint | --> | Interactive Web UI (Satellite Upload, Masks & Contours)   |   |
|   +-----------------------+     +-----------------------------------------------------------+   |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Resource & Checkpoint Verification

### 2.1 Pretrained Checkpoint Specifications

Inspection of the Hugging Face model repository and the author's source code (`Building_Footprint.ipynb`) reveals critical compatibility constraints:

| Specification | Author Checkpoint (`best_deeplab.pth`) | Author Checkpoint (`best_unet.pth`) | Research Paper (Pasquali et al. 2019) |
|---|---|---|---|
| **Base Architecture** | `torchvision.models.segmentation.deeplabv3_resnet50` | Custom 5-stage U-Net | Structurally Optimized U-Net (32-512) |
| **Encoder Backbone** | ResNet-50 | Custom DoubleConv encoder | Custom 4-stage convolutional downsampling |
| **Bottleneck Depth** | ASPP (Atrous Spatial Pyramid Pooling) | 1024 channels | **Removed 1024 layer** (stops at 512) |
| **Batch Normalization**| Included in ResNet backbone | Included (`BatchNorm2d`) | **Omitted** (batch size 2 optimal) |
| **Head Modification** | `classifier[4] = nn.Conv2d(256, 1, 1)` | `nn.Conv2d(64, 1, 1)` | `nn.Conv2d(32, 1, 1)` with Sigmoid |
| **Input Channels** | 3 (RGB Pan-Sharpened) | 3 (RGB Pan-Sharpened) | 8 multispectral or 3 RGB (Section 6) |
| **Input Dimensions** | $512 \times 512$ patches | $512 \times 512$ patches | $384 \times 384$ pixels |
| **Input Normalization**| **Min-Max:** $(x - x_{\min}) / (x_{\max} - x_{\min})$ | **Min-Max:** $(x - x_{\min}) / (x_{\max} - x_{\min})$ | **Z-Score:** $(x - \mu) / \sigma$ per band |
| **Output Classes** | 1 (Binary foreground building logit) | 1 (Binary foreground building logit) | 1 (Binary building probability) |

> [!WARNING]
> **Checkpoint Compatibility Warning:**
> The pretrained `best_deeplab.pth` **cannot** be loaded into `segmentation_models_pytorch.DeepLabV3Plus`. It was built directly on top of `torchvision.models.segmentation.deeplabv3_resnet50`. Loading into an arbitrary SMP implementation will fail with state dictionary key errors.

### 2.2 Model Architecture Class Implementations

```python
# Required PyTorch definitions for checkpoint loading
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabV3Plus(nn.Module):
    """Exact architecture used to save best_deeplab.pth"""
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=None)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
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
    """Exact architecture used to save best_unet.pth"""
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
    """Structurally optimized 32-512 U-Net from Pasquali et al. 2019 (no BatchNorm, 1024 removed)"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.d1 = block(in_channels, 32)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = block(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = block(64, 128)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = block(128, 256)
        self.p4 = nn.MaxPool2d(2)
        self.b = block(256, 512) # Bottleneck at 512, 1024 omitted
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c4 = block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c3 = block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c2 = block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c1 = block(64, 32)
        self.out = nn.Conv2d(32, out_channels, 1)

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
```

---

## 3. Data Leakage Prevention Protocol

The author trained the pretrained checkpoints using the following split:
```python
train_images, val_images, train_geojsons, val_geojsons = train_test_split(
    image_paths, geojson_paths, test_size=0.2, random_state=42
)
```

To maintain an untouched benchmark and avoid data leakage:

1. **Reconstruct the Author's Split:** Run `train_test_split(..., test_size=0.2, random_state=42)` across the complete sorted tile directory.
2. **Designate Unseen Test Partition:** The 20% holdout (validation set in the author's notebook) was **never used for backpropagation/gradient updates** in the author's pretrained weights. 
3. **Partition Segregation:**
   - **Untouched Test Set ($20\%$):** Held strictly for unbiased evaluation of both the baseline pretrained model and our fine-tuned model.
   - **Fine-Tuning Set ($70\%$):** Subdivided from the training partition for training our fine-tuned weights.
   - **Validation Set ($10\%$):** Subdivided from the training partition for early stopping and model checkpoint selection.
4. **Leakage Audit Script:** A verification script (`src/verify_splits.py`) will hash and check all tile IDs, asserting zero intersection between the test partition and any training records.

---

## 4. Phase-by-Phase Implementation Roadmap

### Phase 1: Environment & Dependency Setup
* **Objective:** Configure a reproducible Python environment on Windows with CPU-compatible PyTorch.
* **Dependencies:**
  - `torch`, `torchvision`, `torchaudio` (CPU builds from PyTorch official index)
  - `numpy`, `scipy`, `matplotlib`, `scikit-learn`
  - `tifffile` (for reading multi-band TIFF satellite images)
  - `shapely` (for GeoJSON polygon manipulation and mask conversion)
  - `opencv-python` (for morphological operations, contours, and connected component analysis)
  - `requests` (for downloading checkpoints and dataset tiles)
  - `flask` or `fastapi` (for local visualization web UI)
* **Directory Structure:**
  ```text
  SkyTrace/
  ├── checkpoints/              # Pretrained and fine-tuned .pth files
  │   ├── best_deeplab.pth
  │   └── best_unet.pth
  ├── data/
  │   ├── images/               # Satellite .tif tiles (PS-RGB)
  │   ├── geojson/              # Building footprint .geojson files
  │   ├── masks/                # Generated binary ground truth masks (.png)
  │   └── splits/               # train.json, val.json, test.json
  ├── outputs/
  │   ├── baseline_preds/       # Pretrained visual predictions
  │   ├── finetuned_preds/      # Fine-tuned visual predictions
  │   └── comparison/           # Side-by-side visual comparisons
  ├── src/
  │   ├── models.py             # DeepLabV3Plus, AuthorUNet, PaperOptimizedUNet
  │   ├── dataset.py            # Dataset, patch extraction, augmentations
  │   ├── mask_generator.py     # GeoJSON to binary mask rasterizer
  │   ├── metrics.py            # IoU, Dice, Precision, Recall calculation
  │   ├── train.py              # Fine-tuning training loop & checkpointing
  │   ├── evaluate.py           # Evaluation on untouched test set
  │   └── inference.py          # Sliding window inference & outline overlay
  ├── web/
  │   ├── app.py                # Web server backend
  │   ├── static/               # CSS, JS, UI assets
  │   └── templates/            # HTML templates
  ├── IMPLEMENTATION_PLAN.md    # Reference specification (this document)
  └── run_pipeline.py           # Unified execution runner
  ```

---

### Phase 2: Dataset Acquisition & Mask Generation
* **Objective:** Acquire satellite image tiles and annotations from SpaceNet-2 Paris, and convert GeoJSON polygons into raster masks.
* **Tasks:**
  1. Download representative SpaceNet-2 Paris tiles (`PS-RGB/*.tif` and matching `geojson_buildings/*.geojson`) directly from public SpaceNet S3 storage.
  2. Implement `mask_generator.py` using `shapely.geometry` and scanline rasterization:
     - Map geographic/pixel polygon coordinates to $650 \times 650$ binary raster arrays ($0$ = background, $1$ = building).
  3. Validate mask rasterization visually by overlaying boundaries on true-color RGB tiles.

---

### Phase 3: Train / Validation / Test Splitting & Leakage Audit
* **Objective:** Build deterministic, audited dataset splits.
* **Tasks:**
  1. Sort all acquired tiles deterministically by tile ID.
  2. Execute `train_test_split(image_paths, geojson_paths, test_size=0.2, random_state=42)`.
  3. Save split definitions into `data/splits/train.json`, `data/splits/val.json`, and `data/splits/test.json`.
  4. Run assertion check:
     $$\text{set}(\text{test\_tiles}) \cap (\text{set}(\text{train\_tiles}) \cup \text{set}(\text{val\_tiles})) = \emptyset$$

---

### Phase 4: Model Architecture Binding & Checkpoint Validation
* **Objective:** Download Hugging Face weights and verify strict state dictionary loading.
* **Tasks:**
  1. Download `best_deeplab.pth` and `best_unet.pth` from `https://huggingface.co/Abhatta7/building-footprint-segmentation`.
  2. Instantiate `DeepLabV3Plus` and `AuthorUNet`.
  3. Load weights using `torch.load(..., map_location="cpu")` and verify `load_state_dict(strict=True)`.
  4. Instantiate `PaperOptimizedUNet` (Pasquali et al. 2019 architecture) for comparative fine-tuning.

---

### Phase 5: Baseline Inference & Visual Benchmarks
* **Objective:** Evaluate pretrained models on the untouched test partition before any fine-tuning.
* **Tasks:**
  1. Apply tile min-max normalization:
     $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
  2. Implement sliding-window inference with window size $512 \times 512$ and stride $256$ to process full-resolution tiles.
  3. Generate and save prediction masks and probability heatmaps in `outputs/baseline_preds/`.
  4. Calculate initial baseline metrics on the test partition.

---

### Phase 6: Fine-Tuning Pipeline
* **Objective:** Fine-tune the segmentation network on labeled satellite training images.
* **Tasks:**
  1. **Dataset & Augmentation:**
     - Extract random $512 \times 512$ (or $384 \times 384$) crops containing buildings.
     - Apply spatial augmentations: horizontal flips, vertical flips, and $90^\circ, 180^\circ, 270^\circ$ rotations as specified in Section 4.1 of the paper.
  2. **Loss Function:**
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$
     $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |P \cap Y| + \epsilon}{|P| + |Y| + \epsilon}$$
  3. **Optimization Setup:**
     - Optimizer: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, learning rate $= 10^{-4}$, weight decay $= 10^{-4}$).
     - Learning Rate Scheduler: `ReduceLROnPlateau(factor=0.5, patience=5)`.
     - Early Stopping: Monitor validation IoU with patience of 10 epochs.
  4. **Logging & Checkpointing:**
     - Record per-epoch train loss, validation loss, validation IoU, and Dice score into `logs/training_log.json`.
     - Automatically save `checkpoints/best_finetuned_model.pth`.

---

### Phase 7: Quantitative Evaluation & Metric Comparison
* **Objective:** Compute uncompromised, verifiable metrics comparing pretrained and fine-tuned models on the identical untouched test set.
* **Metrics Computed:**
  1. **Intersection over Union (IoU / Jaccard Index):**
     $$\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
  2. **Dice Coefficient (F1 Score):**
     $$\text{Dice} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
  3. **Precision:**
     $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  4. **Recall:**
     $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  5. **SpaceNet Object-Level F1:** Building polygon matching threshold at $\text{IoU} \ge 0.5$.
* **Deliverable:** Structured comparison table and markdown report (`outputs/evaluation_results.md`).

---

### Phase 8: Interactive Visualization Web Interface
* **Objective:** Provide an intuitive, modern graphical interface to run inference on arbitrary satellite images.
* **Features:**
  1. **Drag-and-Drop Image Uploader:** Accepts `.tif`, `.png`, `.jpg`, and `.geojson`.
  2. **Model Selector:** Choose between Pretrained Baseline, Fine-Tuned Model, or Side-by-Side Comparison.
  3. **Multi-View Visualizer:**
     - Original Satellite Image (True-Color RGB).
     - Predicted Binary Building Mask.
     - Vector Outlines: Polygons/contours overlaid dynamically in high-contrast cyan/amber over the satellite image.
  4. **Telemetry Panel:** Displays building count (via Connected Component Analysis), total estimated building area (pixels/$\text{m}^2$), and confidence distribution.
  5. **Export Tool:** Download predicted masks (.png) and building footprint contours (.geojson).

---

## 5. Execution Verification & Anti-Fabrication Safeguards

To comply with the integrity requirement (*"Do not fabricate metrics or claim that training succeeded unless the code has actually run and the results have been verified"*):

1. Every metric must be computed directly by parsing output arrays from `evaluate.py`.
2. Model weight files on disk must be verified via SHA256 checksums before and after training.
3. Training logs must include actual epoch timestamps and step durations.
4. Visual prediction artifacts must be rendered to disk in `outputs/` and inspected.

---

## 6. Execution Command Reference

| Step | Script Command | Output Artifact |
|---|---|---|
| **1. Install Dependencies** | `pip install -r requirements.txt` | Installed packages |
| **2. Download Dataset & Weights** | `python src/download_resources.py` | `data/` and `checkpoints/` populated |
| **3. Generate Masks & Splits** | `python src/prepare_data.py --random_state 42` | `data/masks/` and `data/splits/` |
| **4. Baseline Evaluation** | `python src/evaluate.py --model_type pretrained` | `outputs/baseline_preds/` |
| **5. Fine-Tune Model** | `python src/train.py --epochs 10 --batch_size 2` | `checkpoints/best_finetuned_model.pth` |
| **6. Comparative Evaluation** | `python src/evaluate.py --compare` | `outputs/evaluation_results.md` |
| **7. Launch Web Application** | `python web/app.py --port 5000` | Local Web UI at `http://localhost:5000` |
