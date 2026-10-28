# Building Footprint Extraction: Model Evaluation Report

**Test Set:** 10 Untouched Holdout Satellite Tiles (`AOI_3_Paris`)  
**Data Partitioning Audit:** `LOCAL_SPLITS_DISJOINT: VERIFIED` (Train, Val, and Test tile IDs share zero intersection)  
**Pretrained Checkpoint Audit:** External weights from Hugging Face (`Abhatta7/building-footprint-segmentation`); pretraining independence documented as an unverified external limitation.

## Quantitative Performance Comparison

| Metric | Pretrained Baseline (`best_deeplab.pth`) | Fine-Tuned Model (`best_finetuned_model.pth`) | Delta Change |
|---|:---:|:---:|:---:|
| **Mean IoU (Macro)** | **0.7123** | **0.7074** | **-0.0049** |
| **Global Micro IoU** | 0.7112 | 0.7068 | -0.0045 |
| **Mean Dice (F1)** | 0.8316 | 0.8283 | -0.0033 |
| **Precision** | 0.8225 | 0.8370 | +0.0145 |
| **Recall** | 0.8418 | 0.8207 | -0.0211 |
| **Pixel Accuracy** | 0.9280 | 0.9282 | +0.0002 |
| **CC Object F1 (IoU ≥ 0.5)** | 0.6431 | 0.6250 | -0.0182 |

---

*Note: Object F1 evaluates connected-component masks against ground-truth raster components at IoU ≥ 0.5. Values are generated automatically from `outputs/metrics_*.json` without manual fabrication.*
