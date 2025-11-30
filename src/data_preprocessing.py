import pandas as pd
import argparse

def load_raw(path):
    return pd.read_csv(path)

def feature_engineer(df):
    df['last_active_date'] = pd.to_datetime(df['last_active_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    today = pd.to_datetime('2025-11-23')

    df['recency_days'] = (today - df['last_active_date']).dt.days
    df['tenure_days'] = (today - df['signup_date']).dt.days

    df['monthly_revenue'] = df['monthly_revenue'].fillna(
        df['monthly_txn_count'] * df['avg_txn_value']
    )

    features = [
        'monthly_txn_count','monthly_revenue','avg_txn_value','recency_days',
        'product_count','is_premium','complaints_last_6m','avg_session_minutes'
    ]
    return df[features + ['churn_3m','customer_id']]

def save_processed(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = load_raw(args.input)
    df2 = feature_engineer(df)
    save_processed(df2, args.output)

    print("Processed file saved:", args.output)