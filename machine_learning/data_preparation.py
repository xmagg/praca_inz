from pathlib import Path
from ucimlrepo import fetch_ucirepo
import pandas as pd


# Ścieżka do głównego katalogu projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIRECTORY / "heart_disease.csv"
VARIABLES_PATH = RAW_DATA_DIRECTORY / "heart_disease_variables.csv"
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIRECTORY / "heart_disease_binary.csv"


def download_heart_disease_data():
    """
    Pobiera zbiór Heart Disease z repozytorium UCI i zapisuje jego lokalną kopię w folderze data/raw
    """
    print("Pobieranie zbioru Heart Disease z UCI...")

    dataset = fetch_ucirepo(id=45)

    # X zawiera 13 cech opisujących pacjentów.
    features = dataset.data.features.copy()

    # y zawiera kolumnę wynikową o nazwie 'num'.
    target = dataset.data.targets.copy()

    # Łączymy cechy oraz wynik w jedną tabelę.
    data = pd.concat([features, target], axis=1)

    RAW_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_DATA_PATH, index=False)
    dataset.variables.to_csv(VARIABLES_PATH, index=False)

    print(f"Dane zapisano w: {RAW_DATA_PATH}")
    print(f"Opis zmiennych zapisano w: {VARIABLES_PATH}")

    return data


def load_raw_data():
    """
    Wczytuje zbiór z lokalnego pliku CSV, jeśli plik nie istnieje albo jest pusty, najpierw pobiera dane
    """
    file_does_not_exist = not RAW_DATA_PATH.exists()
    file_is_empty = (
        RAW_DATA_PATH.exists()
        and RAW_DATA_PATH.stat().st_size == 0
    )

    if file_does_not_exist or file_is_empty:
        return download_heart_disease_data()

    print(f"Wczytywanie danych z: {RAW_DATA_PATH}")

    return pd.read_csv(RAW_DATA_PATH)


def inspect_data(data):
    """Wyświetla podstawowe informacje o pobranym zbiorze"""
    print("\nPierwsze pięć wierszy:")
    print(data.head())

    print("\nLiczba wierszy i kolumn:")
    print(data.shape)

    print("\nNazwy kolumn:")
    print(data.columns.tolist())

    print("\nTypy danych:")
    print(data.dtypes)

    print("\nLiczba brakujących wartości:")
    print(data.isna().sum())

    if "num" in data.columns:
        print("\nRozkład wartości kolumny wynikowej 'num':")
        print(data["num"].value_counts(dropna=False).sort_index())


def prepare_binary_data(data):
    """
    Przygotowuje dane do klasyfikacji binarnej.
    Wartość 0 oznacza brak choroby, natomiast wartości od 1 do 4 zostają zamienione na 1, czyli obecność choroby
    """
    processed_data = data.copy()

    processed_data["target"] = (processed_data["num"] > 0).astype(int)
    processed_data = processed_data.drop(columns=["num"])

    return processed_data


def save_processed_data(data):
    """Zapisuje przygotowane dane w katalogu data/processed"""
    PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nPrzetworzone dane zapisano w: {PROCESSED_DATA_PATH}")


def main():
    raw_data = load_raw_data()

    print("\n=== DANE ORYGINALNE ===")
    inspect_data(raw_data)

    processed_data = prepare_binary_data(raw_data)
    save_processed_data(processed_data)

    print("\n=== DANE PO PRZYGOTOWANIU ===")

    print("\nPierwsze pięć wierszy:")
    print(processed_data.head())

    print("\nRozkład kolumny wynikowej 'target':")
    print(processed_data["target"].value_counts().sort_index())

    print("\nLiczba brakujących wartości:")
    print(processed_data.isna().sum())


if __name__ == "__main__":
    main()