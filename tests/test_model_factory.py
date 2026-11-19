"""Wszystkie fabrycznie tworzone modele potrafią obsłużyć te same dane"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from machine_learning.model_factory import ModelFactory
from machine_learning.train_models import create_model


@pytest.mark.parametrize(
    "name", ["logistic_regression", "decision_tree", "random_forest"]
)
def test_factory_models_fit_and_predict(name):
    path = (
        Path(__file__).resolve().parents[1]
        / "data/processed/heart_disease_binary.csv"
    )
    data = pd.read_csv(path)
    x = data.drop(columns=["target"])
    y = data["target"]
    # Próba zawiera obie klasy; nie zapisuje ani nie podmienia modelu aplikacji
    model = ModelFactory.create_model(name)
    model.fit(x.iloc[:60], y.iloc[:60])
    predictions = model.predict(x.iloc[60:65])
    probabilities = model.predict_proba(x.iloc[60:65])
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (5, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_unknown_model_type_is_reported():
    with pytest.raises(ValueError, match="Nieznany typ klasyfikatora"):
        ModelFactory.create_model("unknown")


def test_final_training_model_keeps_previous_hyperparameters():
    classifier = create_model().named_steps["classifier"]
    assert classifier.C == 1.0
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42
