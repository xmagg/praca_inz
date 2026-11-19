"""Simple Factory: wspólne tworzenie modeli używanych w eksperymentach"""

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]


class ModelFactory:
    """Buduje 3 klasyfikatory z tym samym przetwarzaniem danych"""

    @staticmethod
    def create_classifier(name: str, **parameters) -> ClassifierMixin:
        defaults = {"random_state": 42}
        if name == "logistic_regression":
            classifier_type = LogisticRegression
            defaults["max_iter"] = 1000
        elif name == "decision_tree":
            classifier_type = DecisionTreeClassifier
        elif name == "random_forest":
            classifier_type = RandomForestClassifier
            defaults["n_estimators"] = 100
        else:
            raise ValueError(f"Nieznany typ klasyfikatora: {name}")
        defaults.update(parameters)
        return classifier_type(**defaults)

    @staticmethod
    def create_preprocessor() -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("fill_missing_values", SimpleImputer(strategy="median")),
                ("scaling", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "fill_missing_values",
                    SimpleImputer(strategy="most_frequent"),
                ),
                ("encoding", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, NUMERIC_FEATURES),
                ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ]
        )

    @classmethod
    def create_pipeline(cls, classifier: ClassifierMixin) -> Pipeline:
        return Pipeline(
            steps=[
                ("data_preprocessor", cls.create_preprocessor()),
                ("classifier", classifier),
            ]
        )

    @classmethod
    def create_model(cls, name: str, **parameters) -> Pipeline:
        return cls.create_pipeline(cls.create_classifier(name, **parameters))
