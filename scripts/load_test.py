import random
import sys
import time

import httpx
import pandas as pd

URL = "http://127.0.0.1:8000/predict"
DRIFT = "--drift" in sys.argv

df = pd.read_csv("data/churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df = df.drop(columns=["customerID", "Churn"])

for i, row in enumerate(df.sample(300).to_dict("records")):
    if DRIFT:  # clients récents, factures élevées, contrat mensuel
        row["tenure"] = random.randint(0, 6)
        row["MonthlyCharges"] = round(random.uniform(85, 120), 2)
        row["TotalCharges"] = round(row["tenure"] * row["MonthlyCharges"], 2)
        row["Contract"] = "Month-to-month"
        row["InternetService"] = "Fiber optic"
        row["PaymentMethod"] = "Electronic check"
    r = httpx.post(URL, json=row)
    if i % 50 == 0:
        print(i, r.status_code)
    time.sleep(0.05)