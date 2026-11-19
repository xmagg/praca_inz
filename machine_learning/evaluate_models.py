from pathlib import Path
import pandas as pd

if __package__ in (None, ""):
    from model_factory import ModelFactory, NUMERIC_FEATURES, CATEGORICAL_FEATURES
else:
    from .model_factory import ModelFactory, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_binary.csv"



def load_data():
    """Wczytuje dane i oddziela cechy od kolumny wynikowej"""
    data = pd.read_csv(DATA_PATH)

    x = data.drop(columns=["target"])
    y = data["target"]

    return x, y


def create_preprocessor():
    """Zachowuje dotychczasowy interfejs i wspólne przetwarzanie cech"""
    return ModelFactory.create_preprocessor()


def evaluate_model(
    model_name,
    classifier,
    x_train,
    x_test,
    y_train,
    y_test,
):
    """Uczy jeden model i zwraca wyniki dla zbioru testowego"""
    model = ModelFactory.create_pipeline(classifier)

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    return results


def cross_validate_model(model_name, classifier, x_train, y_train):
    """Przeprowadza 5-krotną walidację krzyżową modelu"""
    model = ModelFactory.create_pipeline(classifier)

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=cross_validation,
        scoring=["accuracy", "f1", "roc_auc"],
    )

    results = {
        "Model": model_name,
        "Accuracy": scores["test_accuracy"].mean(),
        "F1": scores["test_f1"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean(),
    }

    return results


def tune_logistic_regression(x_train, y_train):
    """Porównuje kilka wartości parametru C dla regresji logistycznej"""
    c_values = [0.1, 1, 10]
    results = []

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    for c_value in c_values:
        model = ModelFactory.create_model("logistic_regression", C=c_value)

        scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cross_validation,
            scoring=["accuracy", "f1", "roc_auc"],
        )

        results.append(
            {
                "C": c_value,
                "Accuracy": scores["test_accuracy"].mean(),
                "F1": scores["test_f1"].mean(),
                "ROC-AUC": scores["test_roc_auc"].mean(),
            }
        )

    results_table = pd.DataFrame(results).round(3)

    print("\n=== STROJENIE LOGISTIC REGRESSION ===")
    print(results_table.to_string(index=False))


def main():
    x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Liczba rekordów treningowych: {len(x_train)}")
    print(f"Liczba rekordów testowych: {len(x_test)}")

    models = [
        (
            "Logistic Regression",
            ModelFactory.create_classifier("logistic_regression"),
        ),
        (
            "Decision Tree",
            ModelFactory.create_classifier("decision_tree"),
        ),
        (
            "Random Forest",
            ModelFactory.create_classifier("random_forest"),
        ),
    ]

    test_results = []
    cross_validation_results = []

    for model_name, classifier in models:
        print(f"\nUczenie modelu: {model_name}")

        model_results = evaluate_model(
            model_name,
            classifier,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        test_results.append(model_results)

        cv_results = cross_validate_model(
            model_name,
            classifier,
            x_train,
            y_train,
        )
        cross_validation_results.append(cv_results)

    test_results_table = pd.DataFrame(test_results).round(3)

    cross_validation_table = pd.DataFrame(
        cross_validation_results
    ).round(3)

    print("\n=== WYNIKI NA ZBIORZE TESTOWYM ===")
    print(test_results_table.to_string(index=False))

    print("\n=== 5-KROTNA WALIDACJA KRZYZOWA ===")
    print(cross_validation_table.to_string(index=False))

    tune_logistic_regression(x_train, y_train)


if __name__ == "__main__":
    main()
