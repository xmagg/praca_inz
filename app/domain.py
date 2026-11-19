"""Obiekty danych przekazywane między warstwami aplikacji"""

from dataclasses import asdict, dataclass
import json
from typing import Mapping


PATIENT_FEATURES = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
)


@dataclass(frozen=True, slots=True)
class PatientData:
    """13 cech dla jednego przypadku"""

    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float

    @classmethod
    def from_form(cls, form: Mapping[str, str]) -> "PatientData":
        """Konwertuje formularz z request.form do obiektu PatientData"""
        return cls(
            age=int(form["age"]),
            sex=int(form["sex"]),
            cp=int(form["cp"]),
            trestbps=int(form["trestbps"]),
            chol=int(form["chol"]),
            fbs=int(form["fbs"]),
            restecg=int(form["restecg"]),
            thalach=int(form["thalach"]),
            exang=int(form["exang"]),
            oldpeak=float(form["oldpeak"]),
            slope=int(form["slope"]),
            ca=float(form["ca"]),
            thal=float(form["thal"]),
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def to_json(self) -> str:
        """Reprezentacja do dokumentacji"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Klasa 0/1 i surowe prawdopodobieństwo klasy 1 w przedziale [0, 1]"""

    prediction: int
    probability: float

    def probability_percent(self) -> float:
        """Zachowuje dotychczasowe zaokrąglenie do jednego miejsca"""
        return round(self.probability * 100, 1)

    def get_label(self) -> str:
        return "obecność choroby" if self.prediction == 1 else "brak choroby"

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "prediction": self.prediction,
            "probability": self.probability,
            "probability_percent": self.probability_percent(),
            "label": self.get_label(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
