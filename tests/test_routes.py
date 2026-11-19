from app import create_app


def test_home_page():
    """Sprawdza czy strona główna działa"""
    app = create_app()
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert "Ocena ryzyka".encode("utf-8") in response.data


def test_prediction_page():
    """Sprawdza wysłanie poprawnego formularza"""
    app = create_app()
    client = app.test_client()

    patient_data = {
        "age": "63",
        "sex": "1",
        "cp": "1",
        "trestbps": "145",
        "chol": "233",
        "fbs": "1",
        "restecg": "2",
        "thalach": "150",
        "exang": "0",
        "oldpeak": "2.3",
        "slope": "3",
        "ca": "0",
        "thal": "6",
    }

    response = client.post(
        "/predict",
        data=patient_data,
    )

    assert response.status_code == 200
    assert "Wynik analizy".encode("utf-8") in response.data


def test_invalid_age():
    """Sprawdza odrzucenie nieprawidłowego wieku"""
    app = create_app()
    client = app.test_client()

    patient_data = {
        "age": "10",
        "sex": "1",
        "cp": "1",
        "trestbps": "145",
        "chol": "233",
        "fbs": "1",
        "restecg": "2",
        "thalach": "150",
        "exang": "0",
        "oldpeak": "2.3",
        "slope": "3",
        "ca": "0",
        "thal": "6",
    }

    response = client.post("/predict", data=patient_data)
    assert response.status_code == 200
    assert "Wiek musi mieścić się".encode("utf-8") in response.data
