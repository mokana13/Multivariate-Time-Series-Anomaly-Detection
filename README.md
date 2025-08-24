
# Multivariate-Time-Series-Anomaly-Detection


# Multivariate Time Series Anomaly Detection (Hackathon Submission)

## Overview
This project provides a **Python-based anomaly detection pipeline** for multivariate time series.
It detects anomalies in an industrial dataset, outputs a severity score (0–100), and attributes each anomaly to its top contributing features.

The solution follows the Hackathon specification exactly:
- Learns from a defined **Normal Period** (1/1/2004 00:00 – 1/5/2004 23:59).
- Detects anomalies in an **Analysis Period** (1/6/2004 00:00 – 1/10/2004 07:59).
- Adds **8 new columns** to the dataset:
  - `Abnormality_score`
  - `top_feature_1` … `top_feature_7`

## Deliverables
- `tep_anomaly_detector.py` → Executable script.
- `submission_outline.pdf` → Structured outline (Proposed Solution, Technical Approach, Feasibility, References, Deliverables, Sample Usage).
- `TEP_Train_Test_with_scores.csv` → Example output file with anomaly scores and contributors.
- `README.md` → This file.

## Usage

### Requirements
- Python 3.10+
- Packages: `pandas`, `numpy`, `scikit-learn`

(Optional: `reportlab` if you want to regenerate the outline PDF)

### Run
```bash
python tep_anomaly_detector.py   --input TEP_Train_Test.csv   --output TEP_Train_Test_with_scores.csv   --timestamp-col Time   --train-start "2004-01-01 00:00"   --train-end   "2004-01-05 23:59"   --analysis-start "2004-01-06 00:00"   --analysis-end   "2004-01-10 07:59"   --expected-freq T
```

### Output
- All original columns.
- `Abnormality_score` (0–100).
- `top_feature_1` … `top_feature_7` (names of top contributing features or empty strings if fewer).

## Methodology (Brief)
1. **Data Preprocessing**: Parse timestamps, deduplicate, forward-fill + interpolate missing values, handle constant features, optional frequency validation.
2. **Model Training**: Fit StandardScaler + PCA on Normal Period.
3. **Anomaly Detection**:
   - PCA reconstruction error → captures relationships and pattern deviations.
   - Max absolute z-score vs. training stats → captures threshold violations.
   - Combined robustly, then percentile-ranked to 0–100.
4. **Feature Attribution**: Rank features by residual magnitude, include only contributors >1%, tie-break alphabetically, pad to 7.
5. **Output Generation**: Append 8 new columns.

## Edge Cases Handled
- All normal data (low scores expected).
- Training-period anomalies (warn but proceed).
- Insufficient data (<72 hours) → error.
- Single feature datasets.
- Large datasets (≤10,000 rows).
- Avoids exactly 0 scores.

## Success Criteria
- PEP8-compliant, modular, type-hinted, documented.
- Training period: mean anomaly score <10, max <25.
- Reasonable runtime (≤15 minutes for typical datasets).
- Meaningful feature attribution.

## How to run
Requirements:
pip install -r requirements.txt

Then run:
python src/main.py --input data/TEP_Train_Test.csv --output outputs/TEP_with_scores.csv
This produces outputs/TEP_with_scores.csv with the 8 new columns.

