import json
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

REFERENCE_PATH = "data/churn.csv"
LOG_PATH = "logs/predictions.jsonl"
REPORT_HTML = "reports/drift_report.html"
RESULT_JSON = "reports/drift_result.json"
MIN_ROWS = 100
DRIFT_THRESHOLD = 0.3  # part de colonnes en drift qui déclenche l'alerte


def load_reference():
    df = pd.read_csv(REFERENCE_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df.drop(columns=["customerID", "Churn"]).sample(1000, random_state=42)


def load_current(columns, n=300):
    lines = Path(LOG_PATH).read_text().splitlines()[-n:]
    return pd.DataFrame([json.loads(line) for line in lines])[columns]


def compute_drift():
    reference = load_reference()
    current = load_current(reference.columns.tolist())
    if len(current) < MIN_ROWS:
        raise SystemExit(f"Pas assez de données ({len(current)}/{MIN_ROWS})")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    Path("reports").mkdir(exist_ok=True)
    report.save_html(REPORT_HTML)

    res = report.as_dict()["metrics"][0]["result"]
    share = res["share_of_drifted_columns"]
    out = {
        "share_drifted": share,
        "n_drifted": res["number_of_drifted_columns"],
        "drift_detected": share >= DRIFT_THRESHOLD,
        "n_current": len(current),
    }
    Path(RESULT_JSON).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(compute_drift())