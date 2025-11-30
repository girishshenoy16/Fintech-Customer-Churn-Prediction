import pandas as pd
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def train_and_save(input_csv, output_path):
    df = pd.read_csv(input_csv)

    X = df.drop(columns=['churn_3m','customer_id'])
    y = df['churn_3m']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500))
    ])
    pipe_lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, pipe_lr.predict_proba(X_test)[:,1])

    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    pipe_rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, pipe_rf.predict_proba(X_test)[:,1])

    print("Logistic Regression AUC:", lr_auc)
    print("Random Forest AUC:", rf_auc)

    best = pipe_rf if rf_auc >= lr_auc else pipe_lr
    joblib.dump(best, output_path)
    print("Model saved to:", output_path)
    print("Model Selected:", best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="models/churn_model.pkl")
    args = parser.parse_args()

    train_and_save(args.input, args.output)