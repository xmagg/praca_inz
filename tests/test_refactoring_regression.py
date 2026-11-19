import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from app import create_app
from machine_learning.model_utils import HeartDiseaseModel


ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/behavior_baseline.json").read_text(
        encoding="utf-8"
    )
)

VALIDATION_SCRIPT = re.compile(
    rb'(?m)(?:^[ \t]*\r?\n)?^[ \t]*<script\s+'
    rb'src="/static/js/form-validation\.js"\s*>\s*</script>[ \t]*\r?\n'
)
VALIDATION_STYLES = re.compile(
    rb'(?m)(?:^[ \t]*\r?\n)?^[ \t]*<link\s+rel="stylesheet"\s+'
    rb'href="/static/css/form-validation\.css"\s*/?>[ \t]*\r?\n'
)


@pytest.fixture(scope="module")
def regression_client():
    return create_app({"TESTING": True}).test_client()


@pytest.mark.parametrize(
    "case", BASELINE["http_cases"], ids=lambda case: case["name"]
)
def test_http_response_matches_original(regression_client, case):
    """Pierwotne odpowiedzi z kontrolowaną zmianą zasobów formularza"""
    response = regression_client.open(
        case.get("path", "/predict"),
        method=case.get("method", "POST"),
        data=case.get("data"),
    )
    assert response.status_code == case["status"]
    body = response.data
    if b"<form " in body:
        assert len(VALIDATION_SCRIPT.findall(body)) == 1
        assert len(VALIDATION_STYLES.findall(body)) == 1
        body = VALIDATION_SCRIPT.sub(b"", body, count=1)
        body = VALIDATION_STYLES.sub(b"", body, count=1)
    assert hashlib.sha256(body).hexdigest() == case["body_sha256"]


def test_adapter_preserves_predictions_for_entire_dataset():
    """Klasy i prawdopodobieństwa wszystkich 303 rekordów nie zmieniają się"""
    data = pd.read_csv(
        ROOT / "data/processed/heart_disease_binary.csv"
    ).drop(columns=["target"])
    assert len(data) == BASELINE["dataset_rows"]
    model = HeartDiseaseModel()
    for row in range(len(data)):
        frame = data.iloc[[row]]
        assert model.predict(frame) == BASELINE["predictions"][row]
        assert model.predict_probability(frame) == pytest.approx(
            BASELINE["probabilities"][row], rel=0, abs=1e-12
        )
