"""Interfejs predykcji i adapter do zapisanego pipeline scikit-learn"""

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Wczytuje ten sam artefakt co przed refaktoryzacją"""
    return joblib.load(model_path)


class PredictionModel(ABC):
    """Strategia predykcji, od której zależy serwis aplikacji"""

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> int:
        """Zwraca klasę pierwszego przypadku"""

    @abstractmethod
    def predict_probability(self, data: pd.DataFrame) -> float:
        """Zwraca prawdopodobieństwo klasy 1"""


class HeartDiseaseModel(PredictionModel):
    """Adapter: ukrywa ładowanie joblib oraz tablice wynikowe scikit-learn"""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self._model_path = Path(model_path)
        self.load()

    def load(self) -> None:
        """Ładuje model raz przy tworzeniu obiektu; nie uczy go ponownie"""
        self._pipeline = load_model(self._model_path)

    def predict(self, data: pd.DataFrame) -> int:
        return int(self._pipeline.predict(data)[0])

    def predict_probability(self, data: pd.DataFrame) -> float:
        return float(self._pipeline.predict_proba(data)[0][1])
