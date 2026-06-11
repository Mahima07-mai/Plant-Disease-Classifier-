# 🌿 Plant Pathology — Kaggle Image Classification

A multi-label leaf disease classifier built with PyTorch and a pretrained ResNet-50 backbone, trained with 5-fold cross-validation and evaluated using mean column-wise ROC-AUC (Kaggle's official metric).

---

## 📁 Folder Structure

```
KAGGLE/
├── input/
│   ├── train.csv           # Training labels (one-hot encoded)
│   ├── test.csv            # Test image IDs
│   └── images/             # All .jpg leaf images
├── output/
│   └── sample_submission.csv
├── plant_pathology.py      # Main training & inference script
├── submission.csv          # Generated after training (OOF + test predictions)
└── best_model_fold{n}.pth  # Best model checkpoint per fold (generated at runtime)
```

---

## 🧠 Problem

Multi-label classification of apple leaf images into 4 categories:

| Label              | Description                        |
|--------------------|------------------------------------|
| `healthy`          | No disease present                 |
| `multiple_diseases`| More than one disease present      |
| `rust`             | Rust fungal infection              |
| `scab`             | Apple scab fungal infection        |

Each image is assigned exactly one label (one-hot), but the model outputs independent probabilities per class using `BCEWithLogitsLoss`.

---

## 🏗️ Model Architecture

- **Backbone:** `torchvision` ResNet-50 (ImageNet pretrained)
- **Head:** `Dropout(0.3)` → `Linear(2048 → 4)`
- **Activation at inference:** `Sigmoid` (converts logits to per-class probabilities)

---

## ⚙️ Configuration (`CFG`)

| Parameter      | Value         |
|----------------|---------------|
| `model_name`   | `resnet50`    |
| `img_size`     | `380 × 380`   |
| `num_classes`  | `4`           |
| `epochs`       | `15`          |
| `batch_size`   | `16`          |
| `lr`           | `1e-4`        |
| `weight_decay` | `1e-5`        |
| `n_folds`      | `5`           |
| `seed`         | `42`          |

---

## 🔄 Training Pipeline

1. **Stratified K-Fold** (5 folds) splits based on the argmax label to preserve class balance.
2. For each fold:
   - Train with augmented images (`RandomFlip`, `RandomRotation`, `ColorJitter`)
   - Validate on held-out fold with clean transforms
   - Save the best checkpoint based on validation AUC
   - Generate OOF (out-of-fold) predictions for local AUC scoring
   - Accumulate test predictions (averaged across all folds)
3. **Learning rate schedule:** Cosine Annealing (`eta_min=1e-6`)
4. **Loss:** `BCEWithLogitsLoss`
5. **Metric:** Mean column-wise ROC-AUC (matches Kaggle's leaderboard scoring)

---

## 🖼️ Data Augmentation

| Stage       | Transforms Applied                                              |
|-------------|------------------------------------------------------------------|
| **Train**   | Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(30°), ColorJitter, Normalize |
| **Val/Test**| Resize, Normalize (ImageNet stats)                              |

---

## 📊 Evaluation

After all folds complete, OOF predictions are used to compute per-class and overall AUC:

```
OOF AUC [healthy]:            x.xxxx
OOF AUC [multiple_diseases]:  x.xxxx
OOF AUC [rust]:               x.xxxx
OOF AUC [scab]:               x.xxxx

Overall OOF Mean AUC:         x.xxxx
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch torchvision timm scikit-learn pandas pillow numpy
```

### 2. Prepare input data

Place the following in `input/`:
- `train.csv` — columns: `image_id`, `healthy`, `multiple_diseases`, `rust`, `scab`
- `test.csv` — column: `image_id`
- `images/` — all `.jpg` images (train + test)
- `sample_submission.csv` in `output/`

### 3. Train and generate submission

```bash
python plant_pathology.py
```

This will:
- Train 5 fold models and save the best checkpoint per fold
- Print per-epoch metrics and OOF AUC
- Write `submission.csv` ready for Kaggle upload

---

## 📦 Dependencies

| Package        | Purpose                          |
|----------------|----------------------------------|
| `torch`        | Deep learning framework          |
| `torchvision`  | ResNet backbone & transforms     |
| `timm`         | (imported, available for swap)   |
| `scikit-learn` | StratifiedKFold, ROC-AUC         |
| `pandas`       | CSV handling                     |
| `Pillow`       | Image loading                    |
| `numpy`        | Array operations                 |

---

## 💡 Tips for Improvement

- **Swap backbone:** Replace `resnet50` with `efficientnet_b4` or `convnext_base` via `timm` for better accuracy
- **TTA:** Apply test-time augmentation (horizontal flip) to boost inference AUC
- **Mixup / CutMix:** Add label-mixing augmentation to reduce overfitting
- **Larger image size:** Try `512×512` if VRAM allows
- **Ensemble:** Average predictions from multiple architectures

---

## 📄 License

For Kaggle competition use. Dataset © Kaggle / original authors.
