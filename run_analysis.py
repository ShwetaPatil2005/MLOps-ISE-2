"""
MLflow experiment tracking and simulated data-drift analysis.
Dataset: UCI Online Retail (dataset id 352), downloaded via ucimlrepo.
This script creates a reproducible simulated drift scenario; it does not train a forecasting model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mlflow
from scipy.stats import ks_2samp
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_SEED = 42

def get_dataset():
    dataset = fetch_ucirepo(id=352)
    X = dataset.data.features.copy()
    # Normalize likely UCI column names
    X.columns = [str(c).strip() for c in X.columns]
    # Features commonly returned: InvoiceNo, StockCode, Description, Quantity,
    # InvoiceDate, UnitPrice, CustomerID, Country
    required = ["Quantity", "UnitPrice"]
    missing = [c for c in required if c not in X.columns]
    if missing:
        raise ValueError(f"Expected columns missing from UCI data: {missing}. Found: {list(X.columns)}")
    X["Quantity"] = pd.to_numeric(X["Quantity"], errors="coerce")
    X["UnitPrice"] = pd.to_numeric(X["UnitPrice"], errors="coerce")
    clean = X.dropna(subset=["Quantity", "UnitPrice"]).copy()
    # Exclude returns/cancellations and non-positive prices for a simple demand example.
    clean = clean[(clean["Quantity"] > 0) & (clean["UnitPrice"] > 0)]
    clean = clean[["Quantity", "UnitPrice"]].replace([np.inf, -np.inf], np.nan).dropna()
    return clean.reset_index(drop=True)

def main():
    np.random.seed(RANDOM_SEED)
    data = get_dataset()
    if len(data) < 100:
        raise ValueError("Not enough valid rows after cleaning.")
    split = int(len(data) * 0.7)
    reference = data.iloc[:split].copy()
    current = data.iloc[split:].copy()

    # Controlled drift simulation: change the current-period copy, not the source data.
    # Quantity and unit price are shifted upward to represent a changed demand/price profile.
    simulated = current.copy()
    simulated["Quantity"] = simulated["Quantity"] * 1.50
    simulated["UnitPrice"] = simulated["UnitPrice"] * 1.20

    results = {}
    mlflow.set_experiment("Retail_Demand_Data_Drift")
    with mlflow.start_run(run_name="UCI_Retail_Simulated_Drift"):
        mlflow.log_param("dataset", "UCI Online Retail (id=352)")
        mlflow.log_param("reference_fraction", 0.70)
        mlflow.log_param("simulated_quantity_multiplier", 1.50)
        mlflow.log_param("simulated_unit_price_multiplier", 1.20)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("rows_after_cleaning", int(len(data)))
        mlflow.log_param("reference_rows", int(len(reference)))
        mlflow.log_param("current_rows_before_simulation", int(len(current)))

        for feature in ["Quantity", "UnitPrice"]:
            stat, p_value = ks_2samp(reference[feature].values, simulated[feature].values)
            ref_mean = float(reference[feature].mean())
            cur_mean = float(simulated[feature].mean())
            change_pct = float(((cur_mean - ref_mean) / ref_mean) * 100) if ref_mean else 0.0
            drift_detected = bool(p_value < 0.05)
            results[feature] = {
                "reference_mean": ref_mean,
                "simulated_current_mean": cur_mean,
                "mean_change_percent": change_pct,
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "threshold": 0.05,
                "drift_detected": drift_detected
            }
            mlflow.log_metric(f"{feature.lower()}_reference_mean", ref_mean)
            mlflow.log_metric(f"{feature.lower()}_simulated_mean", cur_mean)
            mlflow.log_metric(f"{feature.lower()}_mean_change_pct", change_pct)
            mlflow.log_metric(f"{feature.lower()}_ks_statistic", float(stat))
            mlflow.log_metric(f"{feature.lower()}_p_value", float(p_value))
            mlflow.log_param(f"{feature.lower()}_drift_detected", str(drift_detected))

        summary = {
            "dataset": "UCI Online Retail (dataset id 352)",
            "source": "https://archive.ics.uci.edu/dataset/352/online+retail",
            "raw_rows_reported_by_source": 541909,
            "rows_after_cleaning": int(len(data)),
            "reference_rows": int(len(reference)),
            "current_rows_before_simulation": int(len(current)),
            "drift_scenario": "Simulated: Quantity x1.50 and UnitPrice x1.20 in current-period copy",
            "detection_method": "Two-sample Kolmogorov-Smirnov test",
            "significance_threshold": 0.05,
            "features": results,
            "interpretation_note": "These results demonstrate simulated feature/data drift, not naturally observed production drift."
        }
        out_json = OUT / "drift_results.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_json))
        reference.to_csv(OUT / "reference_sample.csv", index=False)
        simulated.to_csv(OUT / "simulated_current_sample.csv", index=False)
        mlflow.log_artifact(str(OUT / "reference_sample.csv"))
        mlflow.log_artifact(str(OUT / "simulated_current_sample.csv"))

    print(json.dumps(summary, indent=2))
    print("\nMLflow UI: run `mlflow ui` and open the local URL printed by MLflow.")

if __name__ == "__main__":
    main()
