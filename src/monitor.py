import json
import sys

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.train import load_data

ALPHA = 0.01
DATASET_DRIFT_SHARE = 0.15


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    cols = {}
    for col in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[col]):
            p = ks_2samp(ref[col], cur[col]).pvalue
        else:
            table = pd.concat(
                [ref[col].value_counts(), cur[col].value_counts()], axis=1
            ).fillna(0)
            p = chi2_contingency(table.values)[1]
        cols[col] = {"p_value": float(p), "drift": bool(p < ALPHA)}
    share = sum(c["drift"] for c in cols.values()) / len(cols)
    return {"columns": cols, "drift_share": share,
            "dataset_drift": share > DATASET_DRIFT_SHARE}


def run(simulate=False) -> dict:
    X, _ = load_data()
    ref = X.sample(frac=0.5, random_state=1)
    cur = X.drop(ref.index).copy()
    if simulate:  # simule un changement de population
        cur["MonthlyCharges"] *= 1.5
        cur["tenure"] = (cur["tenure"] * 0.5).astype(int)
        cur["TotalCharges"] = cur["MonthlyCharges"] * cur["tenure"]
        cur["Contract"] = "Month-to-month"
    return drift_report(ref, cur)


if __name__ == "__main__":
    rep = run(simulate="--simulate" in sys.argv)
    drifted = [c for c, v in rep["columns"].items() if v["drift"]]
    print(f"drift_share={rep['drift_share']:.0%} "
          f"dataset_drift={rep['dataset_drift']} colonnes={drifted}")
    with open("drift_report.json", "w") as f:
        json.dump(rep, f, indent=2)