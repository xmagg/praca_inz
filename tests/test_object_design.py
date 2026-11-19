"""Kontrakty obiektów, wymiana strategii i izolacja instancji aplikacji"""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from app import create_app
from app.domain import PATIENT_FEATURES, PatientData, PredictionResult
from app.prediction_service import PredictionService, prepare_patient_data
from app.validation import PatientDataValidationError
from machine_learning.model_utils import PredictionModel


FORM = {
    "age": "63", "sex": "1", "cp": "1", "trestbps": "145",
    "chol": "233", "fbs": "1", "restecg": "2", "thalach": "150",
    "exang": "0", "oldpeak": "2.3", "slope": "3", "ca": "0", "thal": "6",
}


class FixedPredictionModel(PredictionModel):
    """Strategia testowa: nie wymaga pliku joblib ani uczenia"""

    def __init__(self, prediction=1, probability=0.82):
        self.prediction = prediction
        self.probability = probability
        self.calls = 0

    def predict(self, data):
        self.calls += 1
        assert tuple(data.columns) == PATIENT_FEATURES
        return self.prediction

    def predict_probability(self, data):
        self.calls += 1
        return self.probability


def test_patient_data_has_immutable_typed_form_values():
    patient = PatientData.from_form(FORM)
    assert isinstance(patient.age, int)
    assert isinstance(patient.oldpeak, float)
    assert isinstance(patient.ca, float)
    assert tuple(patient.to_dict()) == PATIENT_FEATURES
    assert json.loads(patient.to_json()) == patient.to_dict()
    with pytest.raises(FrozenInstanceError):
        patient.age = 55


def test_frame_preparation_keeps_feature_order_for_both_interfaces():
    patient = PatientData.from_form(FORM)
    object_frame = prepare_patient_data(patient)
    mapping_frame = prepare_patient_data(patient.to_dict())
    assert object_frame.equals(mapping_frame)
    assert object_frame.shape == (1, 13)
    assert tuple(object_frame.columns) == PATIENT_FEATURES


@pytest.mark.parametrize("prediction, probability", [(0, 0.12345), (1, 0.98765)])
def test_result_distinguishes_raw_probability_and_percent(prediction, probability):
    result = PredictionResult(prediction, probability)
    assert result.probability == probability
    assert result.probability_percent() == round(probability * 100, 1)
    assert json.loads(result.to_json()) == result.to_dict()
    assert result.get_label() == (
        "obecność choroby" if prediction else "brak choroby"
    )


def test_invalid_data_is_rejected_before_calling_model():
    model = FixedPredictionModel()
    service = PredictionService(model)
    patient = replace(PatientData.from_form(FORM), age=10, thal=4.0)
    with pytest.raises(PatientDataValidationError) as exception:
        service.predict(patient)
    assert len(exception.value.errors) == 2
    assert model.calls == 0


def test_strategy_can_be_replaced_without_changing_service():
    patient = PatientData.from_form(FORM)
    low = PredictionService(FixedPredictionModel(0, 0.05)).predict(patient)
    high = PredictionService(FixedPredictionModel(1, 0.95)).predict(patient)
    assert low == PredictionResult(0, 0.05)
    assert high == PredictionResult(1, 0.95)


def test_application_instances_keep_separate_dependencies(monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("Wstrzyknięty serwis nie powinien ładować pliku modelu")

    monkeypatch.setattr("machine_learning.model_utils.load_model", unexpected_load)
    low = PredictionService(FixedPredictionModel(0, 0.05))
    high = PredictionService(FixedPredictionModel(1, 0.95))
    first = create_app({"TESTING": True}, prediction_service=low)
    second = create_app({"TESTING": True}, prediction_service=high)
    for app, label, percent in [
        (first, "brak choroby", b"5.0%"),
        (second, "obecność choroby", b"95.0%"),
        (first, "brak choroby", b"5.0%"),
    ]:
        response = app.test_client().post("/predict", data=FORM)
        assert label.encode("utf-8") in response.data
        assert percent in response.data
    assert first.extensions["prediction_service"] is low
    assert second.extensions["prediction_service"] is high


def test_default_app_loads_model_once_for_multiple_requests(monkeypatch):
    from machine_learning import model_utils

    calls = []
    original_load = model_utils.load_model

    def counted_load(path):
        calls.append(path)
        return original_load(path)

    monkeypatch.setattr(model_utils, "load_model", counted_load)
    client = create_app({"TESTING": True}).test_client()
    for _ in range(3):
        assert client.post("/predict", data=FORM).status_code == 200
    assert len(calls) == 1
