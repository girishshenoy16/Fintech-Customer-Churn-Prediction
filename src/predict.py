import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.pkl"

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def predict_single(model, data: dict):
    df = pd.DataFrame([data])
    return model.predict_proba(df)[0][1]

if __name__ == "__main__":
    model = load_model()
    sample = {
        "monthly_txn_count": 5,
        "monthly_revenue": 1000,
        "avg_txn_value": 200,
        "recency_days": 30,
        "product_count": 2,
        "is_premium": 1,
        "complaints_last_6m": 0,
        "avg_session_minutes": 12
    }
    print("Churn Probability:", predict_single(model, sample))