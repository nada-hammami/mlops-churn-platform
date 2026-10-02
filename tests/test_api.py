from fastapi.testclient import TestClient
from src.api.main import app

SAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.0,
}


def test_health_and_predict():
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 200
        assert 0 <= r.json()["churn_probability"] <= 1