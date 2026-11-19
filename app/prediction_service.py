"""Przypadek użycia: walidacja i klasyfikacja danych jednego pacjenta"""

from functools import lru_cache
from typing import Mapping

import pandas as pd

from app.domain import PATIENT_FEATURES, PatientData, PredictionResult
from app.validation import PatientDataValidationError, PatientDataValidator
from machine_learning.model_utils import HeartDiseaseModel, PredictionModel


# Nazwa zachowana dla dotychczasowych importów.
FEATURES = list(PATIENT_FEATURES)


def prepare_patient_data(
    patient_data: PatientData | Mapping[str, int | float],
) -> pd.DataFrame:
    """Przygotowuje ramkę z 13 kolumnami"""
    values = (
        patient_data.to_dict()
        if isinstance(patient_data, PatientData)
        else patient_data
    )
    return pd.DataFrame([values], columns=FEATURES)


class PredictionService:
    """Koordynuje walidator i wstrzykniętą strategię modelu"""

    def __init__(
        self,
        model: PredictionModel,
        validator: PatientDataValidator | None = None,
    ) -> None:
        self._model = model
        self._validator = (
            validator if validator is not None else PatientDataValidator()
        )

    def predict(self, data: PatientData) -> PredictionResult:
        errors = self._validator.validate(data)
        if errors:
            raise PatientDataValidationError(errors)
        frame = prepare_patient_data(data)
        return PredictionResult(
            prediction=self._model.predict(frame),
            probability=self._model.predict_probability(frame),
        )


@lru_cache(maxsize=1)
def _get_default_service() -> PredictionService:
    return PredictionService(HeartDiseaseModel())


def predict_heart_disease(
    patient_data: PatientData | Mapping[str, int | float],
) -> tuple[int, float]:
    data = (
        patient_data
        if isinstance(patient_data, PatientData)
        else PatientData(**patient_data)
    )
    result = _get_default_service().predict(data)
    return result.prediction, result.probability
