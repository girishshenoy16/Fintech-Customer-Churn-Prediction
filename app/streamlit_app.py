import streamlit as st
import pandas as pd
import joblib
import io

model = joblib.load("models/churn_model.pkl")

st.title("Fintech | Churn Prediction & Revenue Impact Simulator")

uploaded = st.file_uploader("Upload processed CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    df['churn_proba'] = model.predict_proba(
        df.drop(columns=['customer_id','churn_3m'], errors="ignore")
    )[:,1]

    st.subheader("Top K High-Risk Customers")
    K = st.slider("Select K", 10, 500, 50)
    top_df = df.sort_values("churn_proba", ascending=False).head(K)
    st.write(top_df[['customer_id','churn_proba','monthly_revenue']])

    st.subheader("Revenue Impact Simulator")
    budget = st.number_input("Budget (₹)", value=50000)
    cost_per_customer = st.number_input("Retention Cost Per Customer (₹)", value=500)
    success_rate = st.number_input("Success Rate (%)", value=40) / 100
    months = st.number_input("Retention Window (months)", value=12)

    max_targetable = int(budget // cost_per_customer)
    candidates = df.sort_values("churn_proba", ascending=False).head(max_targetable)

    expected_saved = candidates.monthly_revenue.sum() * success_rate * months
    roi = (expected_saved - budget) / budget

    st.metric("Expected Saved Revenue", f"₹{expected_saved:,.2f}")
    st.metric("Expected ROI", f"{roi:.2%}")

    buf = io.BytesIO()
    candidates.to_excel(buf, index=False)
    st.download_button("Download Excel", buf, "target_customers.xlsx")
else:
    st.info("Upload the processed CSV to begin.")