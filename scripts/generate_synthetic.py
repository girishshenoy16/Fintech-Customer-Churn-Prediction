import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate(n=5000):
    np.random.seed(42)
    rows = []
    today = datetime(2025,11,23)

    for i in range(n):
        cid = f"CUST_{i:05d}"
        signup = today - timedelta(days=np.random.randint(400, 1500))
        last_active = today - timedelta(days=np.random.randint(1, 120))

        monthly_txn_count = np.random.poisson(3)
        avg_txn = abs(np.random.normal(200, 80))
        monthly_rev = monthly_txn_count * avg_txn
        product_count = np.random.randint(1,4)
        is_premium = np.random.binomial(1, 0.2)
        complaints = np.random.poisson(0.1)
        session = abs(np.random.normal(12,4))

        prob = 0.1 + 0.002*(today - last_active).days - 0.01*monthly_txn_count - 0.05*is_premium
        prob = min(max(prob, 0.02), 0.85)
        churn = np.random.binomial(1, prob)

        rows.append([
            cid, signup.date(), last_active.date(), monthly_txn_count, monthly_rev,
            avg_txn, (today-last_active).days, product_count, is_premium,
            complaints, session, churn
        ])

    df = pd.DataFrame(rows, columns=[
        "customer_id","signup_date","last_active_date","monthly_txn_count",
        "monthly_revenue","avg_txn_value","recency_days","product_count",
        "is_premium","complaints_last_6m","avg_session_minutes","churn_3m"
    ])

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/sample_raw.csv", index=False)
    print("Generated:", "data/raw/sample_raw.csv")

if __name__ == "__main__":
    generate()