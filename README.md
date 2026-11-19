# Aplikacja webowa do oceny ryzyka chorób sercowo-naczyniowych

Projekt realizowany w ramach pracy inżynierskiej.
Aplikacja webowa wykorzystuje model uczenia maszynowego do klasyfikacji danych pacjenta na podstawie wybranych cech klinicznych.
Aplikacja ma charakter edukacyjny i nie stanowi narzędzia diagnostycznego.


## Dane
W projekcie wykorzystano publiczny zbiór danych Heart Disease dostępny w repozytorium UCI Machine Learning Repository.

Dane zawierają informacje dotyczące między innymi:
- wieku,
- płci,
- ciśnienia spoczynkowego,
- poziomu cholesterolu,
- wyniku EKG,
- maksymalnego osiągniętego tętna,
- wybranych parametrów związanych z badaniem serca.

Oryginalna zmienna wynikowa została przekształcona do postaci binarnej:
- 0 – brak choroby,
- 1 – obecność choroby.


## Model uczenia maszynowego
W projekcie porównano następujące algorytmy:
- regresję logistyczną,
- drzewo decyzyjne,
- las losowy.

Do finalnej wersji aplikacji wybrano regresję logistyczną.
Przetwarzanie danych oraz model zostały zapisane jako jeden pipeline biblioteki scikit-learn.


## Technologie
Projekt wykorzystuje:
- Python,
- Flask,
- pandas,
- scikit-learn,
- joblib,
- pytest,
- HTML,
- CSS.


## Uruchomienie projektu
Po pobraniu projektu należy utworzyć i aktywować środowisko wirtualne, a następnie zainstalować wymagane biblioteki:

    pip install -r requirements.txt

Jeżeli przetworzony zbiór danych nie istnieje, należy uruchomić:

    python machine_learning/data_preparation.py

Model można ponownie wytrenować za pomocą:

    python machine_learning/train_models.py

Aplikację webową uruchamia się poleceniem:

    python run.py

Następnie należy otworzyć w przeglądarce:

    http://127.0.0.1:5000


## Testy
Testy można uruchomić poleceniem:

    pytest -v

Testy sprawdzają między innymi:
- poprawność przygotowanych danych,
- działanie predykcji modelu,
- działanie podstawowych endpointów aplikacji Flask.


## Struktura projektu
    cardio_risk/
    ├── app/
    ├── data/
    ├── docs/
    ├── machine_learning/
    ├── models/
    ├── tests/
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    └── run.py


## Informacja
Wynik zwracany przez aplikację ma charakter wyłącznie informacyjny i edukacyjny. Nie stanowi diagnozy medycznej i nie zastępuje konsultacji lekarskiej.


## Projekt obiektowy i wzorce projektowe
Logikę webową podzielono na obiekty danych, kontroler, walidator, serwis
i adapter modelu. W kodzie zastosowano Application Factory, Simple Factory,
Adapter i Strategię. Fabryka modeli jest wspólna dla treningu i porównywania
klasyfikatorów. Formularz i strona wyniku zachowują poprzedni wygląd.


Przed złożeniem rozdziału trzeba wpisać rzeczywisty adres repozytorium i
uzupełnić oznaczone zrzuty GUI. Narzędzie raportujące nie zastępuje pliku
modelu wykorzystywanego przez aplikację.
