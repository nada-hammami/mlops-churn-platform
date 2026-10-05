import pandas as pd
import mlflow,os
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = os.getenv("DATA_PATH", "data/churn.csv")
MODEL_NAME = "churn-model"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])
    return df.drop(columns=["Churn"]), df["Churn"]


def build_pipeline(X, n_estimators, max_depth):
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    return Pipeline([("prep", prep), ("model", model)])


def main(n_estimators=100, max_depth=5):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    mlflow.set_experiment("churn-prediction")
    with mlflow.start_run():
        pipe = build_pipeline(X_train, n_estimators, max_depth)
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        mlflow.log_metrics({"f1": f1, "auc": auc})
        mlflow.sklearn.log_model(
    pipe,
    name="model",
    registered_model_name=MODEL_NAME,
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
        print(f"F1={f1:.3f}  AUC={auc:.3f}")

    return auc

if __name__ == "__main__":
    main()