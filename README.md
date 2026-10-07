# FA26-COMP-671-Yagna

# Land Use Classification from Satellite Images Using Deep Learning
COMP-671 Deep Learning · Final Project · Yagna Saikumar

## Problem
Governments and environmental groups need to know how land is used (forest loss, urban growth, farmland). Manual review across large areas is slow and expensive. This project classifies a 64×64 Sentinel-2 satellite image into one of 10 land-use classes.

**Question:** Does a CNN meaningfully beat a non-deep pixel baseline, and does geometric data augmentation improve generalization?

**Success criteria:** primary metric is test macro-F1. A meaningful result is a CNN ≥ 90 % test accuracy and ≥ 15 points above the baseline.

## Data
- **EuroSAT (RGB)**, Helber et al. (2019), 27,000 labelled images, 10 classes, from the Sentinel-2 satellite.
- Downloaded automatically by `torchvision.datasets.EuroSAT` into `data/`. No manual steps.
- **Split:** 70 / 15 / 15 train/val/test, stratified by class, seed 42. The indices are saved in `results/splits.json`.
- Normalization statistics are computed on the training split only.

## Method
| Model | Description |
|---|---|
| Baseline | Logistic regression on 16×16×3 downsampled pixels |
| CNN_plain | 4-block CNN (Conv-BN-ReLU ×2 + MaxPool), GAP, dropout 0.3 |
| CNN_aug | Same CNN + random flips and 90° rotations (**controlled comparison**) |

## Key hyperparameters (also in `results/config.json`)
seed 42 · batch 128 · 25 epochs · Adam lr 1e-3, weight decay 1e-4 · cosine LR schedule · cross-entropy loss · best checkpoint chosen by validation macro-F1

## Setup & run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Open `land_use_classification_eurosat.ipynb` in VS Code, select the `.venv` kernel, and click **Run All**. With a GPU this takes about 25 minutes. On a CPU, expect a few hours, or set `epochs=5` in `CONFIG` for a quick check.

## Project structure
```
land_use_classification_eurosat.ipynb   # full pipeline: data → baseline → CNN → ablation → evaluation → inference
requirements.txt
models/      # saved weights: CNN_plain.pth, CNN_aug.pth
results/     # splits.json, config.json, results_summary.csv, figures
report/      # final PDF report
```

## Results
| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Baseline (LogReg) | _fill in_ | _fill in_ |
| CNN_plain | _fill in_ | _fill in_ |
| CNN_aug | _fill in_ | _fill in_ |

## Reproducing without retraining
Weights are in `models/`. Run the notebook up through Section 4, then run Section 12 (inference). It loads `models/CNN_aug.pth` directly.

## References
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE JSTARS*, 12(7), 2217–2226.
Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press.









   # ---- COMP-671 project ----
   data/
   *.zip