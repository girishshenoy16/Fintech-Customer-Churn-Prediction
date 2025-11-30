import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score

MODEL = "models/churn_model.pkl"
DATA = "data/processed/train_features.csv"

def evaluate():
    # Load model and data
    model = joblib.load(MODEL)
    df = pd.read_csv(DATA)

    # Prepare X, y
    X = df.drop(columns=['churn_3m', 'customer_id'])
    y = df['churn_3m']

    # Predict probabilities
    y_proba = model.predict_proba(X)[:, 1]

    # ROC Curve computation
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc_score = auc(fpr, tpr)

    # ROC-AUC score (standard)
    roc_auc = roc_auc_score(y, y_proba)

    # Plot ROC
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC (curve) = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig("screenshots/roc_curve.png")
    
    # Print scores
    print("====================================")
    print(f"ROC-AUC Score       : {roc_auc:.4f}")
    print(f"AUC (from curve)    : {auc_score:.4f}")
    print("ROC curve saved to  : screenshots/roc_curve.png")
    print("====================================")

if __name__ == "__main__":
    evaluate()