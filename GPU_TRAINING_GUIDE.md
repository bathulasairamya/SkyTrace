# GPU Training & Multi-System Execution Guide

This guide provides instructions to run model fine-tuning and evaluation on an external system with an **NVIDIA GPU** or via **Google Colab**.

---

## 1. Setup on Another System (NVIDIA GPU Workstation/Laptop)

### Step 1: Clone or Pull the Code
```bash
git clone https://github.com/bathulasairamya/SkyTrace.git
cd SkyTrace
git pull origin main
```

### Step 2: Install PyTorch with CUDA & Dependencies
```bash
# If using a virtualenv or conda:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 3: Fetch Checkpoints & SpaceNet Dataset
```bash
python src/download_resources.py
python src/split_data.py
```

### Step 4: Run GPU-Accelerated Training
```bash
# Trains DeepLabV3+ ResNet-50 with automatic mixed precision on CUDA
python src/train.py --device cuda --epochs 10 --batch_size 4 --lr 0.0001
```
* **VRAM footprint:** ~3.2 GB with batch size 4 and FP16/AMP.
* **Speed:** ~15 to 25 seconds per epoch on an RTX 3060/4060 or T4 GPU.
* The best model will automatically be saved to `checkpoints/best_finetuned_model.pth`.

### Step 5: Evaluate & Compare on Untouched Test Set
```bash
python src/evaluate.py --compare --device cuda
```
* Generates `outputs/evaluation_results.md` with comparative IoU, Dice/F1, Precision, and Recall scores.

### Step 6: Launch Web Interface
```bash
python web/app.py
```
Open `http://localhost:5000` in your browser.

---

## 2. Running for Free on Google Colab (Cloud GPU)

A ready-to-run notebook **`train_gpu.ipynb`** is included directly in the repository!

### Option A: Open `train_gpu.ipynb` directly
1. Go to [Google Colab](https://colab.research.google.com/) and choose **Upload** > upload `train_gpu.ipynb` (or open from GitHub: `bathulasairamya/SkyTrace`).
2. Set Runtime to **T4 GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
3. Click `Runtime` > `Run all`.

### Option B: Run in a fresh Colab notebook
If creating a blank notebook, select **Runtime** > **Change runtime type** > **T4 GPU**, then paste and run:

```python
# Cell 1: Clone repo
!git clone https://github.com/bathulasairamya/SkyTrace.git
%cd SkyTrace

# Cell 2: Install requirements
!pip install -r requirements.txt

# Cell 3: Fetch dataset and pretrained weights
!python src/download_resources.py
!python src/split_data.py

# Cell 4: Train on GPU with Mixed Precision (10 epochs takes ~3 mins)
!python src/train.py --device cuda --epochs 10 --batch_size 4

# Cell 5: Evaluate on untouched test set
!python src/evaluate.py --compare --device cuda

# Cell 6: Download the fine-tuned model checkpoint
from google.colab import files
files.download("checkpoints/best_finetuned_model.pth")
files.download("logs/loss_curve.png")
files.download("logs/training_log.json")
files.download("outputs/evaluation_results.md")
```

---

## 3. Training Script Parameters Reference (`src/train.py`)

| Parameter | Default | Description |
|---|---|---|
| `--model_type` | `deeplabv3` | `deeplabv3`, `author_unet`, or `paper_unet` |
| `--checkpoint` | `checkpoints/best_deeplab.pth` | Base pretrained weights path |
| `--output` | `checkpoints/best_finetuned_model.pth` | Where to save the fine-tuned checkpoint |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` (auto-detects) |
| `--epochs` | `5` | Total training epochs |
| `--batch_size` | `2` | Batch size (use 4 or 8 on GPU) |
| `--lr` | `0.0001` | Learning rate for Adam optimizer |
| `--patch_size` | `512` | Crop patch size |
