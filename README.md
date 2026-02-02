# SMBFT: Spectral-Masked Bidirectional Fusion Transformer

This repository contains the official implementation of **SMBFT** (Spectral‑Masked Bidirectional Fusion Transformer) for hyperspectral image (HSI) classification, aligned with the paper **“Spectral‑Masked Bidirectional Fusion Transformer (SMBFT) for Hyperspectral Image Classification.”**

SMBFT is a dual‑stream architecture that learns **spectral** and **spatial** representations separately and fuses them through **bidirectional cross‑attention** with gated conditioning. An **optional spectral‑masking auxiliary objective** is supported during training. The pipeline emphasizes **reproducibility** with stratified pixel splits, train‑only normalization, optional PCA fit on the training split, and saved split indices.

---

## Highlights

- **Dual‑stream tokenization** (spectral + spatial) with bidirectional cross‑attention.
- **Optional spectral masking (SSL)** for physics‑aligned self‑supervision.
- **Reproducible pipeline**: saved splits, train‑only normalization, PCA state reuse.
- **Automated experimentation**: Optuna hyperparameter tuning, ablations, and reporting.

---

## Repository Structure

```
SMBFT/
├── models/
│   └── final/                 # Trained checkpoints and results
├── scripts/                   # Utility scripts (if any)
├── src/
│   ├── Dataset/               # Dataset folders (Houston13, Pavia_University, Salinas)
│   ├── data/                  # Data loading, preprocessing, augmentations
│   ├── models/                # SMBFT architecture and related models
│   ├── training/              # Train/test/t-SNE/Optuna utilities
│   └── utils/                 # Visualization and helper utilities
├── ablate.py                  # Ablation study runner
├── gather_ablation_results.py # Aggregate ablation metrics and export LaTeX
├── main.py                    # Main CLI (train/test + optional t-SNE)
├── multirun.py                # Multi-seed execution helper
├── summarize_metrics.py       # Aggregate metrics across seeds
└── README.md
```

---

## Datasets

Supported datasets:
- **Houston13**
- **Pavia_University**
- **Salinas**

Place datasets under:
```
src/Dataset/
	├── Houston13/
	├── Pavia_University/
	└── Salinas/
```

Each folder should contain the `.mat` files referenced in [src/data/data_loader.py](src/data/data_loader.py). A helper downloader is available in [src/data/downloader.py](src/data/downloader.py).

---

## Installation

Create a Python environment (conda or venv) and install requirements:

```
pip install -r requirements.txt
```

---

## Quick Start

### Train
```
python main.py --mode train --dataset Houston13 --output_dir models/final/smbft/Houston13
```

### Test
```
python main.py --mode test --dataset Houston13 --output_dir models/final/smbft/Houston13 \
	--checkpoint models/final/smbft/Houston13/smbft_best_Houston13.pth
```

### t‑SNE (from checkpoint)
```
python src/training/tsne_plot.py --dataset Houston13 \
	--checkpoint models/final/smbft/Houston13/smbft_best_Houston13.pth --force
```

---

## Reproducibility Details (Matches the Paper)

- **Stratified pixel split** with saved indices (train/val/test).
- **Normalization** uses train split statistics only.
- **PCA (optional)** is fit on train pixels and applied to the full cube; the PCA state is saved in checkpoints and reused for testing and visualization.

---

## Ablation Studies

Run ablations on a dataset:
```
python ablate.py --dataset Houston13
```

Aggregate results and export LaTeX:
```
python gather_ablation_results.py --dataset Houston13
```

The LaTeX table is saved to:
```
reports/results/ablation_results_Houston13.tex
```

---

## Hyperparameter Tuning (Optuna)

Run two‑stage coarse‑to‑fine search:
```
python src/training/smbft_hyperparameter_tuning.py
```

Generate Optuna reports/plots:
```
python src/training/generate_optuna_reports.py --dataset Houston13 --db_dir reports/optuna_db
```

---

## Results & Reporting

Summarize metrics across seeds:
```
python summarize_metrics.py
```

Outputs are written to:
```
reports/results/
	├── metrics_summary_full.csv
	├── metrics_summary_full.txt
	└── metrics_summary_latex.txt
```

---

## Key Modules

- **Model**: [src/models/model_architecture.py](src/models/model_architecture.py)
- **Training**: [src/training/train.py](src/training/train.py)
- **Testing**: [src/training/test.py](src/training/test.py)
- **t‑SNE**: [src/training/tsne_plot.py](src/training/tsne_plot.py)
- **Data & PCA**: [src/data/data_loader.py](src/data/data_loader.py), [src/utils/utils.py](src/utils/utils.py)

---

---

## Citation

If you find this work useful in your research, please consider citing:

```BibTeX
@Article{Keskin2026f,
  author   = {Fesih Keskin},
  journal  = {},
  title    = {},
  year     = {2026},
  issn     = {},
  pages    = {},
  doi      = {},
  keywords = {},
  url      = {},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

The authors gratefully acknowledge the providers of the Houston 2013, Pavia University, and Salinas hyperspectral datasets for making these benchmarks publicly available.