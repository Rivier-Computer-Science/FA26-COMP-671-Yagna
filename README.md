# FA26-COMP-671-Yagna

# Land Use Classification from Satellite Images Using Deep Learning
COMP-671 Deep Learning · Final Project · Yagna Saikumar

## Problem
Governments and environmental groups need to know how land is used (forest loss, urban growth, farmland). Manual review across large areas is slow and expensive. This project classifies a 64×64 Sentinel-2 satellite image into one of 10 land-use classes.

**Question:** Does a CNN meaningfully beat a non-deep pixel baseline, and does geometric data augmentation improve generalization?

**Success criteria:** primary metric is test macro-F1. A meaningful result is a CNN ≥ 90 % test accuracy and ≥ 15 points above the baseline. Augmentation counts as helping if it raises macro-F1 by more than ~0.5 points.

## Data
- **EuroSAT (RGB)**, Helber et al. (2019), 27,000 labelled images, 10 classes, from the Sentinel-2 satellite.
- Downloaded automatically by `torchvision.datasets.EuroSAT` into `data/`. No manual steps.
- **Split:** 70 / 15 / 15 train/val/test, stratified by class, seed 42 (18,900 / 4,050 / 4,050 images). The notebook saves the indices to `results/splits.json` so the split is identical on every run.
- Normalization statistics are computed on the training split only.

## Method
| Model | Description |
|---|---|
| Baseline | Logistic regression on 16×16×3 downsampled pixels |
| CNN_plain | 4-block CNN (Conv-BN-ReLU ×2 + MaxPool), global average pooling, dropout 0.3, 1.18 M parameters |
| CNN_aug | Same CNN + random flips and 90° rotations (**controlled comparison**: only the training transform changes) |

## Key hyperparameters
seed 42 · batch 128 · 25 epochs · Adam lr 1e-3, weight decay 1e-4 · cosine LR schedule · cross-entropy loss · best checkpoint chosen by validation macro-F1. All settings (data split, model, training, augmentation, seeds) are documented in `config.yaml`; the notebook also writes them to `results/config.json`.

## Setup & run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Or, with conda: `conda env create -f environment.yml` then `conda activate eurosat-landuse`.

Open `land_use_classification_eurosat.ipynb` in VS Code or Jupyter, select the `.venv` kernel, and click **Run All**.

Training was run on a **Google Colab T4 GPU**: about 15 minutes for Sections 1–13, plus about 30 minutes for the 3-seed check in Section 14. The notebook also runs on CPU (it picks the device automatically), but expect several hours; set `epochs=5` in `CONFIG` for a quick check.

## Project structure
```
land_use_classification_eurosat.ipynb   # full pipeline: data → baseline → CNN → ablation → evaluation → inference → 3-seed check
requirements.txt                        # pip dependencies
environment.yml                         # conda environment (pinned versions)
config.yaml                             # all experiment settings
results/     # figures, results_summary.csv, multiseed_runs.csv, multiseed_summary.csv
report/      # final PDF report
```

## Results
Held-out test set (4,050 images). Single run with seed 42:

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Baseline (LogReg on pixels) | 44.7 % | 0.439 |
| CNN_plain (no augmentation) | 97.3 % | 0.972 |
| CNN_aug (flips + 90° rotations) | **98.0 %** | **0.979** |

**Robustness check, 3 seeds (42, 1, 2), mean ± std:**

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| CNN_plain | 97.36 ± 0.19 % | 0.9727 ± 0.0020 |
| CNN_aug | **97.87 ± 0.13 %** | **0.9781 ± 0.0015** |

**Key findings**
- The CNN beats the pixel baseline by **53 points**, so spatial feature learning is essential.
- Augmentation improves macro-F1 by **+0.53 points on average and wins in 3/3 seeds**. It also reduces overfitting: final training loss is 0.060 vs 0.024 without augmentation, at similar validation F1.
- Errors concentrate in the agricultural classes (AnnualCrop ↔ PermanentCrop ↔ Pasture), which look alike in single-date RGB imagery.

Figures in `results/`: `class_distribution.png`, `samples.png`, `learning_curves.png`, `confusion_matrices.png`, `misclassified.png`, `confidence.png`, `inference_demo.png`.

## Model weights / retraining
Trained weights are not stored in the repository. To reproduce the reported models, run the notebook top to bottom: the fixed seed and saved split recreate the same setup. Small run-to-run differences (about ±0.2 points) are expected on GPU; Section 14 measures this spread. Training saves the weights to `models/CNN_plain.pth` and `models/CNN_aug.pth`, and Section 12 (inference) loads them from there.

## References
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2217–2226.

Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press.
