from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import sys
from time import perf_counter_ns
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from app import create_app
from app.domain import PATIENT_FEATURES, PatientData
from machine_learning.model_factory import ModelFactory
from machine_learning.model_utils import HeartDiseaseModel


def metrics(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions)),
        "recall": float(recall_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def cv_summary(pipeline, x_train, y_train):
    scores = cross_validate(
        pipeline, x_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=["accuracy", "f1", "roc_auc"],
    )
    return {
        name: {
            "mean": float(scores["test_" + name].mean()),
            "std": float(scores["test_" + name].std(ddof=0)),
            "folds": scores["test_" + name].tolist(),
        }
        for name in ["accuracy", "f1", "roc_auc"]
    }


def junit_summary(path):
    if not path.exists():
        return {"available": False, "command": "python -m pytest -q --junitxml=docs/rozdzial4/wyniki/pytest.xml"}
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    groups = {}
    for case in cases:
        name = case.get("classname", "").split(".")[-1] + ".py"
        group = groups.setdefault(name, {"total": 0, "passed": 0, "failed": 0, "skipped": 0})
        group["total"] += 1
        if case.find("failure") is not None or case.find("error") is not None:
            group["failed"] += 1
        elif case.find("skipped") is not None:
            group["skipped"] += 1
        else:
            group["passed"] += 1
    return {
        "available": True,
        "total": len(cases),
        "passed": sum(g["passed"] for g in groups.values()),
        "failed": sum(g["failed"] for g in groups.values()),
        "skipped": sum(g["skipped"] for g in groups.values()),
        "time_seconds": sum(float(s.get("time", 0)) for s in root.iter("testsuite")),
        "files": groups,
    }


def measure(operation, repetitions):
    for _ in range(10):
        operation()
    samples = []
    for _ in range(repetitions):
        start = perf_counter_ns()
        operation()
        samples.append((perf_counter_ns() - start) / 1_000_000)
    return {
        "warmup": 10, "repetitions": repetitions, "unit": "ms",
        "mean": float(np.mean(samples)), "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)), "min": float(min(samples)),
        "max": float(max(samples)),
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/rozdzial4/wyniki")
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("Liczba powtórzeń musi być dodatnia.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / "data/processed/heart_disease_binary.csv"
    model_path = ROOT / "models/heart_disease_pipeline.joblib"
    original_model_hash = sha256(model_path.read_bytes()).hexdigest()
    data = pd.read_csv(data_path)
    x, y = data.drop(columns=["target"]), data["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )

    comparison = []
    cross_validation = []
    refitted_lr = None
    for key, label in [
        ("logistic_regression", "Regresja logistyczna"),
        ("decision_tree", "Drzewo decyzyjne"),
        ("random_forest", "Las losowy"),
    ]:
        pipeline = ModelFactory.create_model(key)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        comparison.append({"model": label, **metrics(y_test, predictions, probabilities)})
        cross_validation.append({"model": label, "scores": cv_summary(ModelFactory.create_model(key), x_train, y_train)})
        if key == "logistic_regression":
            refitted_lr = pipeline

    tuning = []
    for c in [0.1, 1.0, 10.0]:
        scores = (
            cross_validation[0]["scores"] if c == 1.0
            else cv_summary(ModelFactory.create_model("logistic_regression", C=c), x_train, y_train)
        )
        tuning.append({"C": c, "scores": scores})

    adapter = HeartDiseaseModel()
    saved_predictions = adapter._pipeline.predict(x_test)
    saved_probabilities = adapter._pipeline.predict_proba(x_test)[:, 1]
    deployed_metrics = metrics(y_test, saved_predictions, saved_probabilities)
    fpr, tpr, _ = roc_curve(y_test, saved_probabilities)

    app = create_app({"TESTING": True})
    client = app.test_client()
    service = app.extensions["prediction_service"]
    functional = []
    for number, (_, row) in enumerate(data.iloc[:5].iterrows(), 1):
        form = {
            name: format(float(row[name]), "g") if name == "oldpeak" else str(int(row[name]))
            for name in PATIENT_FEATURES
        }
        result = service.predict(PatientData.from_form(form))
        response = client.post("/predict", data=form)
        rendered = [v.strip() for v in re.findall(r"<strong>([^<]+)</strong>", response.get_data(as_text=True))]
        expected = [result.get_label(), f"{result.probability_percent()}%"]
        assert response.status_code == 200 and rendered == expected
        functional.append({
            "id": f"FT-{number:02}", "form": form,
            "prediction": result.prediction, "probability": result.probability,
            "probability_percent": result.probability_percent(), "label": result.get_label(),
            "http_status": response.status_code, "passed": True,
        })

    baseline_form = functional[0]["form"]
    invalid_specs = [
        ("VAL-01", "age", "3", "Wiek musi mieścić się w zakresie 18-100 lat."),
        ("VAL-02", "age", "101", "Wiek musi mieścić się w zakresie 18-100 lat."),
        ("VAL-03", "chol", None, "Wprowadź poprawne wartości we wszystkich polach."),
        ("VAL-04", "oldpeak", "abc", "Wprowadź poprawne wartości we wszystkich polach."),
        ("VAL-05", "thal", "4", "Nieprawidłowy wynik badania thal."),
        ("VAL-06", "oldpeak", "nan", "Wartość oldpeak musi mieścić się w zakresie 0-7."),
    ]
    invalid = []
    for case_id, field, value, message in invalid_specs:
        form = dict(baseline_form)
        if value is None:
            form.pop(field)
        else:
            form[field] = value
        with patch.object(service._model, "predict") as predict, patch.object(service._model, "predict_probability") as probability:
            response = client.post("/predict", data=form)
            assert response.status_code == 200
            assert message in response.get_data(as_text=True)
            assert predict.call_count == probability.call_count == 0
        invalid.append({"id": case_id, "field": field, "value": value, "message": message, "passed": True})

    patient = PatientData.from_form(baseline_form)
    benchmark = {
        "service": measure(lambda: service.predict(patient), args.repetitions),
        "flask_test_client": measure(lambda: client.post("/predict", data=baseline_form), args.repetitions),
        "scope": "Jeden proces; sekwencyjnie; model już w pamięci. Klient testowy nie mierzy sieci ani przeglądarki.",
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
                        "libraries": {name: version(name) for name in ["Flask", "pandas", "scikit-learn", "joblib", "pytest", "numpy"]}},
        "data": {"rows": len(data), "features": len(PATIENT_FEATURES), "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
                 "train_rows": len(x_train), "test_rows": len(x_test),
                 "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
                 "missing_values": {k: int(v) for k, v in x.isna().sum().items()}, "sha256": sha256(data_path.read_bytes()).hexdigest()},
        "model": {"sha256": original_model_hash, "classes": adapter._pipeline.classes_.tolist(), "classifier": type(adapter._pipeline.named_steps["classifier"]).__name__,
                  "C": adapter._pipeline.named_steps["classifier"].C, "metrics": deployed_metrics,
                  "confusion_matrix": confusion_matrix(y_test, saved_predictions, labels=[0, 1]).tolist(),
                  "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                  "refitted_predictions_equal": bool(np.array_equal(saved_predictions, refitted_lr.predict(x_test))),
                  "refitted_probability_max_difference": float(np.max(np.abs(saved_probabilities - refitted_lr.predict_proba(x_test)[:, 1])))},
        "comparison": comparison, "cross_validation": cross_validation, "logistic_tuning": tuning,
        "pytest": junit_summary(output / "pytest.xml"), "functional_cases": functional,
        "server_validation_cases": invalid, "benchmark": benchmark,
        "gui": {"live_browser_verified_this_run": False, "note": "Zrzuty bieżącego GUI należy wykonać w uruchomionej aplikacji."},
    }
    assert sha256(model_path.read_bytes()).hexdigest() == original_model_hash
    (output / "raport_implementacji.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(output / "porownanie_modeli.csv", comparison)
    write_csv(output / "przypadki_funkcjonalne.csv", [
        {"id": c["id"], **c["form"], "prediction": c["prediction"], "probability_percent": c["probability_percent"], "label": c["label"]}
        for c in functional
    ])
    print(json.dumps({"report": str(output / "raport_implementacji.json"), "pytest": report["pytest"], "comparison": comparison,
                      "deployed_metrics": deployed_metrics, "confusion_matrix": report["model"]["confusion_matrix"],
                      "benchmark": benchmark, "model_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
