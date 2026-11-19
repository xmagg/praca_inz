from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import json
import joblib
import pandas as pd

if __package__ in (None, ""):
    from model_factory import ModelFactory, NUMERIC_FEATURES, CATEGORICAL_FEATURES
else:
    from .model_factory import ModelFactory, NUMERIC_FEATURES, CATEGORICAL_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_binary.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"



def load_data():
    """Wczytuje dane i oddziela cechy od kolumny wynikowej"""
    data = pd.read_csv(DATA_PATH)

    x = data.drop(columns=["target"])
    y = data["target"]

    return x, y


def create_model():
    """Tworzy niezmieniony finalny pipeline przez wspólną fabrykę"""
    return ModelFactory.create_model("logistic_regression", C=1.0)


def calculate_metrics(model, x_test, y_test):
    """Oblicza podstawowe metryki dla zbioru testowego"""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 3),
        "precision": round(precision_score(y_test, predictions), 3),
        "recall": round(recall_score(y_test, predictions), 3),
        "f1_score": round(f1_score(y_test, predictions), 3),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 3),
    }

    return metrics, predictions


def save_metrics(metrics):
    """Zapisuje wyniki modelu do pliku JSON"""
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def main():
    x, y = load_data()

    # 80% danych służy do uczenia modelu
    # a 20% do jego końcowego sprawdzenia
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Liczba rekordów treningowych: {len(x_train)}")
    print(f"Liczba rekordów testowych: {len(x_test)}")

    model = create_model()

    # Przygotowanie danych i uczenie finalnego modelu
    model.fit(x_train, y_train)

    metrics, predictions = calculate_metrics(
        model,
        x_test,
        y_test,
    )

    print("\n=== WYNIKI FINALNEGO MODELU ===")

    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1-score:  {metrics['f1_score']}")
    print(f"ROC-AUC:   {metrics['roc_auc']}")

    print("\nMacierz pomyłek:")
    print(confusion_matrix(y_test, predictions))

    # Tworzy folder models, jeżeli jeszcze nie istnieje
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Zapis wytrenowanego modelu
    joblib.dump(model, MODEL_PATH)

    # Zapis wyników modelu
    save_metrics(metrics)

    print(f"\nModel zapisano w: {MODEL_PATH}")
    print(f"Metryki zapisano w: {METRICS_PATH}")


if __name__ == "__main__":
    main()
