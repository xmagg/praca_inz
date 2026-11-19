"""Walidacja domenowa niezależna od Flask i od wybranego modelu ML"""

from typing import Mapping

from app.domain import PatientData


class PatientDataValidationError(ValueError):
    """Przekazuje wszystkie błędy walidacji do kontrolera formularza"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__(" ".join(self.errors))


class PatientDataValidator:
    """Sprawdza niezmienione zakresy i kody kategorii formularza"""

    def validate(
        self, data: PatientData | Mapping[str, int | float]
    ) -> list[str]:
        values = data.to_dict() if isinstance(data, PatientData) else data
        errors = []

        if not 18 <= values["age"] <= 100:
            errors.append("Wiek musi mieścić się w zakresie 18-100 lat.")
        if not 80 <= values["trestbps"] <= 220:
            errors.append(
                "Ciśnienie spoczynkowe musi mieścić się w zakresie 80-220 mm Hg."
            )
        if not 100 <= values["chol"] <= 600:
            errors.append(
                "Poziom cholesterolu musi mieścić się w zakresie 100-600 mg/dl."
            )
        if not 60 <= values["thalach"] <= 220:
            errors.append(
                "Maksymalne tętno musi mieścić się w zakresie 60-220 uderzeń/min."
            )
        if not 0 <= values["oldpeak"] <= 7:
            errors.append("Wartość oldpeak musi mieścić się w zakresie 0-7.")
        if values["sex"] not in [0, 1]:
            errors.append("Nieprawidłowa wartość pola płeć.")
        if values["cp"] not in [1, 2, 3, 4]:
            errors.append("Nieprawidłowy typ bólu w klatce piersiowej.")
        if values["fbs"] not in [0, 1]:
            errors.append("Nieprawidłowa wartość pola poziomu cukru.")
        if values["restecg"] not in [0, 1, 2]:
            errors.append("Nieprawidłowy wynik EKG.")
        if values["exang"] not in [0, 1]:
            errors.append("Nieprawidłowa wartość pola dławicy wysiłkowej.")
        if values["slope"] not in [1, 2, 3]:
            errors.append("Nieprawidłowa wartość nachylenia odcinka ST.")
        if values["ca"] not in [0.0, 1.0, 2.0, 3.0]:
            errors.append("Nieprawidłowa liczba głównych naczyń.")
        if values["thal"] not in [3.0, 6.0, 7.0]:
            errors.append("Nieprawidłowy wynik badania thal.")

        return errors
