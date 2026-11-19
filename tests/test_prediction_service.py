from app.prediction_service import predict_heart_disease


def test_prediction():
    """Sprawdza, czy model zwraca klasę i prawdopodobieństwo."""

    patient_data = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0.0,
        "thal": 6.0,
    }

    prediction, probability = predict_heart_disease(patient_data)

    assert prediction in [0, 1]
    assert 0 <= probability <= 1
