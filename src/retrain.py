import os
import sys

import mlflow
import requests
from mlflow import MlflowClient

from src.monitor import run as check_drift
from src.train import MODEL_NAME, main as train

API_URL = os.getenv("API_URL", "http://localhost:8000")


def main(simulate=False):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()

    if not check_drift(simulate)["dataset_drift"]:
        print("Pas de drift : rien à faire.")
        return

    prod = client.get_model_version_by_alias(MODEL_NAME, "production")
    old_auc = client.get_run(prod.run_id).data.metrics["auc"]

    new_auc = train()
    new_v = max(client.search_model_versions(f"name='{MODEL_NAME}'"),
                key=lambda m: int(m.version)).version

    if new_auc >= old_auc:
        client.set_registered_model_alias(MODEL_NAME, "production", new_v)
        print(f"v{new_v} promu (AUC {old_auc:.3f} -> {new_auc:.3f})")
        try:
            requests.post(f"{API_URL}/reload", timeout=10)
        except requests.RequestException:
            print("API injoignable : relance-la pour charger le nouveau modèle.")
    else:
        print(f"v{new_v} refusé (AUC {new_auc:.3f} < {old_auc:.3f})")


if __name__ == "__main__":
    main(simulate="--simulate" in sys.argv)