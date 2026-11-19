"""Kontroler formularza oraz niezmienione adresy i nazwy widoków Flask"""

from typing import Mapping

from flask import Blueprint, current_app, render_template, request

from app.domain import PatientData
from app.prediction_service import PredictionService
from app.validation import PatientDataValidationError, PatientDataValidator


main_blueprint = Blueprint("main", __name__)


class PredictionController:
    """Tłumaczy formularz HTTP na dane domenowe i wynik na widok HTML"""

    def __init__(self, prediction_service: PredictionService) -> None:
        self._prediction_service = prediction_service

    def index(self) -> str:
        return render_template("index.html")

    def predict(self, form_data: Mapping[str, str]) -> str:
        try:
            patient_data = PatientData.from_form(form_data)
        except (ValueError, KeyError):
            return render_template(
                "index.html",
                error="Wprowadź poprawne wartości we wszystkich polach.",
                form_data=form_data,
            )
        try:
            result = self._prediction_service.predict(patient_data)
        except PatientDataValidationError as error:
            return render_template(
                "index.html",
                error=" ".join(error.errors),
                form_data=form_data,
            )
        return render_template(
            "result.html",
            prediction=result.prediction,
            probability=result.probability_percent(),
        )


def validate_patient_data(patient_data):
    return PatientDataValidator().validate(patient_data)


@main_blueprint.route("/")
def index():
    return current_app.extensions["prediction_controller"].index()


@main_blueprint.route("/predict", methods=["POST"])
def predict():
    return current_app.extensions["prediction_controller"].predict(request.form)
