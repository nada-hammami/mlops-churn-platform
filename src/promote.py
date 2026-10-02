import sys
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
version = sys.argv[1]
MlflowClient().set_registered_model_alias("churn-model", "production", version)
print(f"churn-model v{version} -> alias 'production'")