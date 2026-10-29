# ISE 2 — MLflow Installation, Experiment Tracking and Data Drift Analysis

## Use case
Retail demand monitoring using the public UCI Online Retail dataset. This is an independent academic experiment in the inventory / retail domain; it is not a claim that SmartShelfX itself trained a model on this dataset.

## Dataset
UCI Online Retail, dataset ID 352: https://archive.ics.uci.edu/dataset/352/online+retail  
The UCI page reports 541,909 transactions and includes Quantity, InvoiceDate, StockCode and UnitPrice among its fields. The script downloads it through `ucimlrepo`, cleans positive quantity/price rows, and uses Quantity and UnitPrice for a drift demonstration.

## Environment setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_analysis.py
mlflow ui
```
Open the local MLflow UI address printed by the command (commonly `http://127.0.0.1:5000`).

## What the experiment does
1. Downloads UCI Online Retail dataset.
2. Keeps positive Quantity and UnitPrice rows for this simplified example.
3. Splits records into a 70% reference portion and 30% current portion.
4. Simulates drift in a copy of the current portion by multiplying Quantity by 1.50 and UnitPrice by 1.20.
5. Applies a two-sample Kolmogorov–Smirnov (KS) test to each feature; p < 0.05 is the detection criterion.
6. Logs parameters, metrics, and artifacts to MLflow and writes `outputs/drift_results.json`.

## Important interpretation
The drift is deliberately simulated to validate the workflow. Do not describe it as naturally observed or as evidence that the production SmartShelfX system is drifting. The KS test detects distribution differences; statistical significance does not by itself prove business impact.

## Outputs
`outputs/drift_results.json`, `outputs/reference_sample.csv`, and `outputs/simulated_current_sample.csv` are generated when the script runs. Use the exact results JSON and the MLflow UI to complete Section 4 of the assignment. Do not invent results before running the script.

## Reproducibility
- Fixed seed: 42
- Dataset source: UCI Machine Learning Repository, dataset 352
- Drift configuration: Quantity × 1.50; UnitPrice × 1.20
- Detection threshold: p < 0.05
- MLflow experiment: `Retail_Demand_Data_Drift`

## GitHub
Create a repository, for example `mlflow-retail-data-drift-ise2`, and upload this folder. Add the repository URL to the report after you publish it. Do not claim a GitHub URL until the repository actually exists.
