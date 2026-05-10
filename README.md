# GNSS / ADS-B Spoofing Detection Dashboard

<img width="1882" height="900" alt="Image" src="https://github.com/user-attachments/assets/71ba2e9b-7759-4b10-8ce2-0e8ceb493c37" />

<img width="1887" height="902" alt="Image" src="https://github.com/user-attachments/assets/34d0c4d1-e337-4a91-afd5-27bf29e46a0f" />

Projekt przedstawia system do wykrywania anomalii oraz potencjalnych ataków spoofingu w danych nawigacyjnych. Aplikacja łączy dwa podejścia detekcji:

1. **Model ADS-B** — analizuje trajektorie lotów oraz parametry wiadomości ADS-B.
2. **Model GNSS-SDR** — analizuje cechy sygnału radiowego GNSS wyekstrahowane z danych przetworzonych przez GNSS-SDR.

System umożliwia wizualizację lotów na mapie, prezentację podejrzanych przypadków, porównanie skuteczności modeli oraz analizę cech wykorzystywanych przez modele uczenia maszynowego.

---

## Cel projektu

Celem projektu jest pokazanie, że spoofing sygnałów nawigacyjnych można analizować na dwóch poziomach:

- **poziomie trajektorii** — np. nagły skok pozycji, nienaturalna zmiana prędkości, kursu lub wysokości,
- **poziomie sygnału radiowego** — np. zmiany C/N0, Dopplera, błędów PLL/DLL, asymetrii korelatorów i dominacji prompt.

Dzięki połączeniu obu podejść system pozwala lepiej zrozumieć problem spoofingu GNSS oraz anomalii w danych lotniczych.

---

## Główne funkcje

- Wykrywanie anomalii w danych ADS-B.
- Wykrywanie spoofingu GNSS na podstawie cech sygnału radiowego.
- Interaktywna mapa live z danymi OpenSky Network.
- Panel lotów o wysokim prawdopodobieństwie anomalii.
- Statyczne mapy demonstracyjne pokazujące odchylenia trajektorii.
- Dashboard porównujący dwa modele detekcji.
- Wizualizacja metryk modeli:
  - accuracy,
  - precision,
  - recall,
  - F1-score,
  - ROC AUC,
  - confusion matrix.
- Wizualizacja ważności cech modelu.
- Porównanie różnych progów decyzyjnych modelu.
- Obsługa dashboardów HTML generowanych z danych i wyników modeli.

---

## Zastosowane technologie

### Python

Projekt został napisany w języku Python, ponieważ posiada rozbudowany ekosystem bibliotek do analizy danych, uczenia maszynowego, przetwarzania sygnałów oraz wizualizacji.

### FastAPI

Warstwa webowa została przygotowana z użyciem frameworka FastAPI. Pozwala on łatwo tworzyć endpointy API, serwować widoki HTML oraz integrować aplikację z kodem analitycznym napisanym w Pythonie.

### Uvicorn

Uvicorn służy do uruchamiania aplikacji FastAPI jako lokalnego serwera ASGI.

### pandas i NumPy

Biblioteki `pandas` i `numpy` zostały użyte do:

- wczytywania danych CSV,
- czyszczenia danych,
- przygotowania cech,
- agregacji danych,
- przetwarzania zbiorów treningowych i testowych.

### scikit-learn

Do budowy modeli uczenia maszynowego wykorzystano bibliotekę `scikit-learn`. W projekcie zastosowano algorytm Random Forest, ponieważ dobrze sprawdza się na danych tabelarycznych i pozwala analizować ważność cech.

### GNSS-SDR

GNSS-SDR został wykorzystany jako źródło cech sygnału radiowego GNSS. Na podstawie danych trackingowych uzyskano cechy takie jak:

- C/N0,
- Doppler,
- PLL lock,
- błędy śledzenia nośnej,
- błędy śledzenia kodu,
- asymetria korelatorów early/late,
- dominacja prompt.

### Folium

Biblioteka `folium` została użyta do generowania interaktywnych map HTML opartych o OpenStreetMap. Mapy pokazują trajektorie, punkty normalne, punkty spoofowane oraz podejrzane odchylenia.

### Plotly

`Plotly` zostało wykorzystane do tworzenia interaktywnych wykresów w dashboardzie, m.in. wykresów metryk, macierzy pomyłek, ważności cech oraz rozkładów prawdopodobieństw predykcji.

### joblib

`joblib` służy do zapisu i odczytu wytrenowanych modeli oraz listy cech używanych przez model.

### OpenSky Network API

OpenSky Network API zostało wykorzystane do pobierania aktualnych danych ADS-B o lotach. Dane te pozwalają generować mapę live oraz testować model ADS-B na danych zbliżonych do rzeczywistych.

---

## Struktura projektu

Przykładowa struktura katalogów:

```text
.
├── app.py
├── opensky_worker.py
├── requirements.txt
├── Dockerfile
├── models/
│   ├── adsb_attack_detector_trajectory.pkl
│   └── model_features_trajectory.pkl
├── data/
│   ├── opensky_history.csv
│   ├── opensky_latest_predictions.csv
│   ├── real_gps_track.csv
│   └── model_results/
│       ├── adsb/
│       │   ├── model_summary.json
│       │   ├── confusion_matrix.csv
│       │   ├── feature_importance.csv
│       │   └── threshold_results.csv
│       └── gnss_sdr/
│           ├── model_summary.json
│           ├── confusion_matrix.csv
│           ├── feature_importance.csv
│           ├── threshold_results.csv
│           ├── cv_results.csv
│           └── holdout_predictions.csv
├── features/
│   ├── all_gnss_window_features.csv
│   ├── random_forest_gnss_spoofing_grouped_v2.joblib
│   ├── feature_importance_grouped_v2.csv
│   └── grouped_evaluation_report_v2.txt
├── scripts/
│   ├── generate_models_dashboard.py
│   ├── prepare_av_gps_dataset.py
│   └── generate_bin_signal_dashboard.py
├── static/
│   ├── generated/
│   │   ├── opensky_live_map.html
│   │   ├── model_dashboard.html
│   │   └── bin_signal_comparison.html
│   └── archived_maps/
│       └── av_gps_spoofing_map.html
└── templates/
    ├── layout.html
    ├── map.html
    ├── static_maps.html
    ├── model_dashboard.html
    └── bin_dashboard.html
