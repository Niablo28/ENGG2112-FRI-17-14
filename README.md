# ENGG2112 – Sleep Quality Predictor (Group Project)

## Quick Start

```bash
# 1) Set up environment
py -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
py -m pip install -r requirements.txt

# 2) Place data
# Put the real dataset at: data/raw/kaggle_sleephealth.csv
# We include data/sample.csv so you can run UI without the full dataset.

# 3) Train (CV + OOF + calibrated threshold)
py scripts\retrain_end_to_end.py
# Or: make train (if make is available)

# 4) Generate figures for the report
py scripts\threshold_sweep.py
py scripts\calibration_eval.py
py scripts\fairness_slices.py
# Or: make figures (if make is available)

# 5) Smoke tests
py -m pytest tests\ -q
# Or: make test (if make is available)

# 6) Run the UI
py -m streamlit run ui\app.py
# Or: make ui (if make is available)
```

## What We Submit

* `submission/ENGG2112_Final_Report.pdf` (main document)
* Code (this repo) with one final model `models/model_YYYYMMDD_SHA.joblib`
* Figures under `reports/figures/` referenced in the report
* `reports/METRICS.md` (concise metrics table)
* `models/model_card.md` (intended use, data, metrics, limits, ethics)

## Project Structure

```
ENGG2112-FRI-17-14/
├─ scripts/
│  ├─ retrain_end_to_end.py     # Main training script (use this)
│  ├─ threshold_sweep.py
│  ├─ calibration_eval.py
│  ├─ fairness_slices.py
│  ├─ check_leakage.py
│  └─ deprecated/                # Legacy scripts (hard-stopped)
├─ models/
│  └─ model_YYYYMMDD_SHA.joblib  # Final trained model
├─ reports/
│  ├─ figures/                   # Generated visualizations
│  └─ METRICS.md                 # Performance metrics
├─ ui/
│  └─ app.py                     # Streamlit demo app
└─ data/
   └─ sample.csv                 # Sample data for testing
```

## Reproducibility Notes

* Deterministic seeds set in CV & models (`random_state=42`)
* No test leakage; shuffle-label AUC ≈ 0.5 by `make check-leakage`
* Ablation included (no-synthetic vs augmented) – see report §Results
* All preprocessing inside pipeline during CV (prevents leakage)

## Key Features

- **End-to-end pipeline:** Preprocessing inside CV prevents data leakage
- **OOF threshold selection:** Threshold optimized on out-of-fold predictions
- **Calibration:** ECE/Brier metrics and reliability diagrams
- **Fairness evaluation:** Performance slices by gender/age
- **Leakage checks:** Shuffle-label test and duplicate detection

## Make Targets

- `make train` - Train model with proper CV
- `make figures` - Generate all report figures
- `make test` - Run pytest tests
- `make ui` - Launch Streamlit app
- `make check-leakage` - Verify no data leakage
- `make clean` - Remove intermediate files
- `make package` - Create submission bundle
