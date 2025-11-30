from src.predict import load_model, predict_single

def test_prediction_output():
    model = load_model()
    sample = {
        "monthly_txn_count": 3,
        "monthly_revenue": 600,
        "avg_txn_value": 200,
        "recency_days": 20,
        "product_count": 2,
        "is_premium": 0,
        "complaints_last_6m": 0,
        "avg_session_minutes": 10
    }
    prob = predict_single(model, sample)
    assert 0 <= prob <= 1