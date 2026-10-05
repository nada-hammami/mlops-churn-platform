import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
import json
from pathlib import Path
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@production")

state = {}
LOG_PATH = Path(os.getenv("PRED_LOG", "logs/predictions.jsonl"))
LOG_PATH.parent.mkdir(exist_ok=True)

def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

PREDICTIONS = Counter(
    "churn_predictions_total", "Nombre de prédictions", ["outcome"]
)
PROBA = Histogram(
    "churn_probability", "Distribution des probabilités de churn",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# L'ordre des champs = l'ordre des colonnes à l'entraînement
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(customer: Customer):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([customer.model_dump()])
    proba = float(state["model"].predict_proba(df)[0, 1])
    PREDICTIONS.labels(outcome="churn" if proba >= 0.5 else "no_churn").inc()
    PROBA.observe(proba)
    with LOG_PATH.open("a") as f:

        f.write(json.dumps(customer.model_dump()) + "\n")
    return {"churn_probability": round(proba, 4), "churn": proba >= 0.5}


@app.post("/reload")
def reload_model():
    load_model()
    return {"status": "reloaded"}