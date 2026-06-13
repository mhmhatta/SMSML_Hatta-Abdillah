# Credit Card Fraud Detection MSML Runbook

Root project:

```text
E:\projects\Dicoding\MSML\SMSML_Muhammad-Hatta-Abdillah
```

Main dataset:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

Expected raw file:

```text
Membangun_model/data_raw/creditcard.csv
```

## 1. Local Environment

Use the existing virtual environment from the Dicoding workspace:

```powershell
cd E:\projects\Dicoding\MSML\SMSML_Muhammad-Hatta-Abdillah
E:\projects\Dicoding\.venv311\Scripts\Activate.ps1
python -m pip install -r .\Membangun_model\requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
E:\projects\Dicoding\.venv311\Scripts\Activate.ps1
```

## 2. Dataset Setup

For local runs, put the downloaded Kaggle CSV here:

```text
Membangun_model/data_raw/creditcard.csv
```

Current project already has this file. If starting from zero, download it manually from Kaggle and keep the filename exactly `creditcard.csv`.

Do not commit the full CSV unless the submission rule explicitly allows it. The `.gitignore` excludes it by default.

## 3. DagsHub + MLflow Remote Tracking

For this project, use DagsHub only as a **remote MLflow dashboard**.

You do **not** need to upload `creditcard.csv` to DagsHub. Dataset upload/DVC is optional and can make the project ribet for no real scoring benefit. What Dicoding needs here is proof that experiment tracking works: params, metrics, and artifacts appear in DagsHub/MLflow.

Your current DagsHub repo:

```text
https://dagshub.com/mhmhatta/smsml
```

So the MLflow tracking URI is:

```text
https://dagshub.com/mhmhatta/smsml.mlflow
```

### Step 1 - Create DagsHub Token

In DagsHub:

```text
Profile icon -> User Settings -> Tokens -> Generate New Token
```

Copy the token once. Treat it like a password.

### Step 2 - Set Environment Variables in PowerShell

Run this in the same terminal before training:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/mhmhatta/smsml.mlflow"
$env:MLFLOW_TRACKING_USERNAME="mhmhatta"
$env:MLFLOW_TRACKING_PASSWORD="<paste-your-dagshub-token-here>"
```

Check:

```powershell
echo $env:MLFLOW_TRACKING_URI
echo $env:MLFLOW_TRACKING_USERNAME
```

Do not commit the token. Do not write it into Python files.

### Step 3 - Run Training from That Same Terminal

```powershell
cd E:\projects\Dicoding\MSML\SMSML_Muhammad-Hatta-Abdillah\Membangun_model
python cc_preprocessing\preprocess.py
python modelling.py
python modelling_tuning.py --n-iter 4 --cv 3 --max-rows 80000
```

For non-interactive environments like GitHub Actions, do not use `dagshub.init()` OAuth. The scripts use MLflow environment variables directly.

Open this after training:

```text
https://dagshub.com/mhmhatta/smsml/experiments
```

or directly:

```text
https://dagshub.com/mhmhatta/smsml.mlflow
```

You should see runs named:

```text
baseline_logistic_regression
tuned_random_forest
```

### Step 4 - If Running from Notebook

PowerShell env vars may not reach an already-running notebook kernel. If you run MLflow cells from the notebook, add this temporary cell near the top, run it once, then run the modelling cells:

```python
import os

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/mhmhatta/smsml.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "mhmhatta"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<paste-your-dagshub-token-here>"
```

Remove the token from the notebook before submitting. Screenshot first if needed, then delete the token cell output too.

### Step 5 - Evidence to Capture

Take screenshots of:

- DagsHub `Experiments` tab showing both runs
- tuned model metrics: `average_precision`, `roc_auc`, `recall`, `precision`, `f1_score`
- model artifact page
- confusion matrix / feature importance artifacts if visible

If you do not set DagsHub variables, MLflow logs locally under `mlruns/`. That still works technically, but for advanced criteria remote DagsHub evidence is better.

## 4. Run the Notebook Experiment

Open this notebook:

```text
Eksperimen_SML_Muhammad-Hatta-Abdillah.ipynb
```

Run it from the project root:

```text
E:\projects\Dicoding\MSML\SMSML_Muhammad-Hatta-Abdillah
```

This notebook follows the provided `Template_Eksperimen_MSML.ipynb` structure, then adds baseline modelling, tuning, and MLflow logging.

## 5. Run the Script Pipeline

Use this when you want clean reproducible artifacts outside the notebook:

```powershell
cd E:\projects\Dicoding\MSML\SMSML_Muhammad-Hatta-Abdillah\Membangun_model
python cc_preprocessing\preprocess.py
python modelling.py
python modelling_tuning.py --n-iter 8 --cv 3
```

Fast CI-style run:

```powershell
python modelling_tuning.py --n-iter 4 --cv 3 --max-rows 80000
```

Expected outputs:

```text
Membangun_model/cc_preprocessing/processed/train.csv
Membangun_model/cc_preprocessing/processed/test.csv
Membangun_model/artifacts/model.joblib
Membangun_model/artifacts/amount_time_scaler.joblib
Membangun_model/artifacts/feature_columns.json
```

## 6. Run with MLflow Project

Use the local virtual environment:

```powershell
cd E:\projects\Dicoding\MSML\SMSML_Muhammad-Hatta-Abdillah\Membangun_model
mlflow run . --env-manager=local -e full_pipeline -P raw_path=data_raw/creditcard.csv -P processed_dir=cc_preprocessing/processed -P n_iter=4 -P cv=3 -P max_rows=80000
```

Use larger tuning for final evidence:

```powershell
mlflow run . --env-manager=local -e full_pipeline -P raw_path=data_raw/creditcard.csv -P processed_dir=cc_preprocessing/processed -P n_iter=8 -P cv=3 -P max_rows=0
```

## 7. Inspect MLflow Locally

If using local MLflow:

```powershell
cd E:\projects\Dicoding\MSML\SMSML_Muhammad-Hatta-Abdillah
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

If using DagsHub, inspect the run from the DagsHub MLflow dashboard instead.

## 8. Run API Serving + Monitoring

Make sure the final model artifacts exist first:

```text
Membangun_model/artifacts/model.joblib
Membangun_model/artifacts/amount_time_scaler.joblib
Membangun_model/artifacts/feature_columns.json
```

Start Docker stack:

```powershell
cd E:\projects\Dicoding\MSML\SMSML_Muhammad-Hatta-Abdillah\Monitoring_dan_Logging
docker compose up --build
```

Open:

```text
API health:  http://localhost:8000/health
API metrics: http://localhost:8000/metrics
Prometheus:  http://localhost:9090
Grafana:     http://localhost:3000
```

Grafana login:

```text
admin / admin
```

Send one test prediction:

```powershell
$body = Get-Content .\sample_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body $body
```

Generate more traffic for Grafana:

```powershell
1..20 | ForEach-Object {
  Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body $body
}
```

## 9. GitHub Actions Config

Workflow file:

```text
.github/workflows/train.yml
```

GitHub repository secrets needed:

```text
KAGGLE_USERNAME
KAGGLE_KEY
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

Kaggle token source:

```text
Kaggle Account -> Settings -> API -> Create New Token
```

The workflow will:

1. install dependencies
2. download `creditcard.csv` from Kaggle if missing
3. run preprocessing
4. train baseline model
5. train tuned model
6. log MLflow metrics/artifacts

Important: GitHub Actions only detects `.github/workflows/train.yml` if this folder is the repository root. If your GitHub repo root is `E:\projects\Dicoding`, move/copy the `.github` folder to that root or make `MSML/SMSML_Muhammad-Hatta-Abdillah` the repo root.

## 10. Evidence Checklist for Submission

Capture these screenshots:

- notebook cells showing EDA and preprocessing
- MLflow/DagsHub experiment page
- tuned model metrics
- model artifact page
- GitHub Actions successful workflow
- FastAPI `/predict` response
- Prometheus targets page
- Prometheus metrics query
- Grafana dashboard
- Grafana alert rule

Submission folders already exist for monitoring evidence:

```text
Monitoring_dan_Logging/1.bukti_serving
Monitoring_dan_Logging/4.bukti monitoring Prometheus
Monitoring_dan_Logging/5.bukti monitoring Grafana
Monitoring_dan_Logging/6.bukti alerting Grafana
```

## 11. Common Issues

### `IProgress not found`

Harmless Jupyter warning. Fix:

```powershell
python -m pip install -U ipywidgets jupyter
```

Restart the notebook kernel.

### API fails on startup

Usually artifacts are missing. Run:

```powershell
cd E:\projects\Dicoding\MSML\SMSML_Muhammad-Hatta-Abdillah\Membangun_model
python modelling_tuning.py
```

### CI cannot find dataset

Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in GitHub Secrets. Do not put Kaggle credentials in the repo.

### DagsHub does not receive runs

Check:

```powershell
echo $env:MLFLOW_TRACKING_URI
echo $env:MLFLOW_TRACKING_USERNAME
```

Then rerun training from the same terminal session.
