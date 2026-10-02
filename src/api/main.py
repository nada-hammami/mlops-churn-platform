import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@production")

state = {}


def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


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
    return {"churn_probability": round(proba, 4), "churn": proba >= 0.5}


@app.post("/reload")
def reload_model():
    load_model()
    return {"status": "reloaded"}