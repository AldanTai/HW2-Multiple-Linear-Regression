# HW2 Multiple Linear Regression — Step 5

## Problems
1. Student Performance Factors — target `Exam_Score`, original dataset: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
2. King County House Sales — target `price`, original dataset: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

**50 Startups is not included.** Both problems can use a local built-in CSV, user uploaded CSV, or direct publicly accessible HTTPS CSV URL. If a user uploads a different dataset, they can pick the numeric target and exclude ID/leakage columns.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Place these downloaded CSVs into `data/`:
- `StudentPerformanceFactors.csv`
- `kc_house_data.csv`

Alternatively use sidebar **上傳 CSV** to analyze without placing local files. Kaggle dataset description URLs are NOT direct CSV URLs; download the CSV from Kaggle first. Public raw GitHub CSV links can be provided under **輸入 CSV URL**.

## Workflow
- 80/20 outer split, then 75/25 fit/calibration.
- CV selects K on model-fit subset only.
- Split-conformal 95% interval uses calibration samples; observed coverage on test samples.
- Pipeline fits preprocessing and feature selection only on training folds.
- Avoid ID and target leakage; original King County id is dropped; sale date converted into year/month.

## Deployment (Streamlit Community Cloud)
Push project to GitHub (without `.venv/`); deploy repository with entrypoint `app.py`. For built-in mode, commit public dataset files if licenses allow, or use uploaded CSV and HTTPS links instead. Do not commit private CSVs. Verify app from another device / incognito tab before submission; Streamlit free hosting may sleep.

## Safety / Scope
Public HTTPS CSV URLs only (no redirects, local/private addresses). Uploaded CSV size limit is 12 MB in app, and 40–50,000 rows; high-cardinality text features must be removed. This is a classroom demo, not a hardened enterprise ingestion service. Predictions and confidence intervals depend on the training distribution and modeling assumptions. The 95% interval is a marginal prediction interval, not a confidence interval for the conditional mean.
