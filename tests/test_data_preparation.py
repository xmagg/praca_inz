from pathlib import Path
from machine_learning.data_preparation import prepare_binary_data
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_binary.csv"


def test_processed_data_exists():
    """Sprawdza czy przetworzony plik z danymi istnieje"""
    assert DATA_PATH.exists()


def test_target_column_exists():
    """Sprawdza czy zbiór zawiera kolumnę wynikową target"""
    data = pd.read_csv(DATA_PATH)

    assert "target" in data.columns


def test_target_contains_only_two_classes():
    """Sprawdza czy target zawiera tylko klasy 0 i 1"""
    data = pd.read_csv(DATA_PATH)
    target_values = set(data["target"].unique())

    assert target_values == {0, 1}


def test_prepare_binary_data():
    """Sprawdza poprawne przekształcenie wartości num na klasy 0 i 1"""
    data = pd.DataFrame(
        {
            "age": [50, 55, 60, 65, 70],
            "num": [0, 1, 2, 3, 4],
        }
    )

    processed_data = prepare_binary_data(data)

    assert processed_data["target"].tolist() == [0, 1, 1, 1, 1]
    assert "num" not in processed_data.columns
