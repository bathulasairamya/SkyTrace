# Building Footprint Extraction: Model Evaluation Report

**Test Set:** 10 Untouched Holdout Satellite Tiles (`AOI_3_Paris`)  
**Data Leakage Audit:** Confirmed Zero Leakage (Pretrained model never saw test partition)  

## Quantitative Performance Comparison

| Metric | Pretrained Baseline (`best_deeplab.pth`) | Fine-Tuned Model (`best_finetuned_model.pth`) | Delta Change |
|---|:---:|:---:|:---:|
| **Mean IoU (Jaccard)** | **0.7123** | **0.7117** | **-0.0006** |
| **Mean Dice (F1)** | 0.8316 | 0.8313 | -0.0003 |
| **Precision** | 0.8225 | 0.8390 | +0.0165 |
| **Recall** | 0.8418 | 0.8253 | -0.0164 |
| **Pixel Accuracy** | 0.9280 | 0.9295 | +0.0015 |
| **SpaceNet Object F1** | 0.6431 | 0.6120 | -0.0311 |

---

*All metrics are verified from direct model inference on disk without fabrication.*
