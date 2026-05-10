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
```
---

## Sposób uruchomienia kodu

Poniżej znajduje się przykładowa kolejność uruchamiania projektu lokalnie.

---

### 1. Instalacja zależności

Najpierw utwórz i aktywuj środowisko wirtualne:

```bash
python -m venv venv
```
Windows:
```
venv\Scripts\activate
```
Linux / macOS:
```
source venv/bin/activate
```
Następnie zainstaluj zależności:
```
pip install -r requirements.txt
```
2. Przygotowanie wyników modeli

Jeżeli dashboard modeli nie został jeszcze wygenerowany, należy najpierw uruchomić skrypty treningowe lub skrypty przygotowujące wyniki modeli.

Model ADS-B

Model ADS-B analizuje dane trajektorii i wiadomości ADS-B.

Uruchom:
```
python model_pipeline.py
```
Skrypt powinien zapisać między innymi:
```
models/adsb_attack_detector_trajectory.pkl
models/model_features_trajectory.pkl
data/model_results/adsb/model_summary.json
data/model_results/adsb/confusion_matrix.csv
data/model_results/adsb/feature_importance.csv
data/model_results/adsb/threshold_results.csv
Model GNSS-SDR
```
Model GNSS-SDR analizuje cechy sygnału radiowego GNSS uzyskane z GNSS-SDR.

Uruchom:
```
python train_model_grouped_v2.py
```
Skrypt powinien zapisać między innymi:
```
features/random_forest_gnss_spoofing_grouped_v2.joblib
data/model_results/gnss_sdr/model_summary.json
data/model_results/gnss_sdr/confusion_matrix.csv
data/model_results/gnss_sdr/feature_importance.csv
data/model_results/gnss_sdr/threshold_results.csv
data/model_results/gnss_sdr/cv_results.csv
data/model_results/gnss_sdr/holdout_predictions.csv
```
3. Generowanie dashboardu modeli

Po przygotowaniu wyników modeli należy wygenerować dashboard HTML:
```
python scripts/generate_models_dashboard.py
```
Wynik zostanie zapisany do:
```
static/generated/model_dashboard.html
```
Dashboard będzie dostępny w aplikacji pod adresem:
```
http://127.0.0.1:8000/model-dashboard
```
4. Generowanie statycznej mapy spoofingu GPS/GNSS

Aby wygenerować statyczną mapę pokazującą przykładowe odchylenie trajektorii GPS/GNSS:
```
python scripts/prepare_av_gps_dataset.py
```
Skrypt zapisze pliki:
```
data/real_gps_track.csv
static/archived_maps/av_gps_spoofing_map.html
```
Mapa będzie widoczna w aplikacji pod adresem:
```
http://127.0.0.1:8000/maps
```
5. Generowanie dashboardu sygnałów BIN

Jeżeli dostępne są pliki binarne z sygnałami GNSS, np.:
```
clear_sky_5000mb.bin
spoof_1_5000mb.bin
spoof_2_5000mb.bin
spoof_4_5000mb.bin
```
można wygenerować dashboard porównujący sygnały naturalne i spoofowane:
```
python scripts/generate_bin_signal_dashboard.py
```
Wynik zostanie zapisany do:
```
static/generated/bin_signal_comparison.html
```

Dashboard będzie dostępny pod adresem:
```
http://127.0.0.1:8000/bin-dashboard
```
6. Uruchomienie aplikacji webowej

Aplikację FastAPI uruchamiamy poleceniem:

uvicorn app:app --host 127.0.0.1 --port 8000

Po uruchomieniu aplikacja będzie dostępna pod adresem:

http://127.0.0.1:8000
7. Uruchomienie mapy live OpenSky

Mapa live korzysta z danych OpenSky Network API. Worker pobiera aktualne dane ADS-B, wykonuje predykcję modelem ADS-B i generuje mapę HTML.

Worker może być uruchomiony na dwa sposoby.

Opcja A — uruchomienie workera osobno

W osobnym terminalu uruchom:
```
python opensky_worker.py
```
Worker zapisuje wyniki do:
```
data/opensky_history.csv
data/opensky_latest_predictions.csv
static/generated/opensky_live_map.html
```
Mapa live będzie dostępna pod adresem:

http://127.0.0.1:8000/map
Opcja B — uruchomienie workera razem z aplikacją

W pliku app.py ustaw:

ENABLE_OPENSKY_WORKER = True

Następnie uruchom aplikację:
```
uvicorn app:app --host 127.0.0.1 --port 8000
```
Na czas developmentu można ustawić:
```
ENABLE_OPENSKY_WORKER = False
```
Jest to przydatne, ponieważ OpenSky API posiada rate limity.

8. Najważniejsze adresy aplikacji

Po uruchomieniu aplikacji dostępne są następujące widoki:
```
http://127.0.0.1:8000
```
Strona główna.
```
http://127.0.0.1:8000/map

Mapa live OpenSky z predykcjami anomalii.

http://127.0.0.1:8000/maps

Statyczne mapy demonstracyjne.

http://127.0.0.1:8000/model-dashboard
```
Dashboard porównujący model ADS-B i model GNSS-SDR.
```
http://127.0.0.1:8000/bin-dashboard
```
Dashboard porównujący sygnały naturalne i spoofowane.

9. Endpointy API

Aplikacja udostępnia również proste endpointy API.

Status aplikacji
GET /api/status

Przykład:
```
http://127.0.0.1:8000/api/status
```
Zwraca informacje o stanie aplikacji, dostępności mapy, pliku predykcji oraz workera OpenSky.

Podsumowanie predykcji
GET /api/predictions/summary

Przykład:

http://127.0.0.1:8000/api/predictions/summary

Zwraca podstawowe statystyki aktualnych predykcji modelu ADS-B.

Lista predykcji
GET /api/predictions

Przykład:
```
http://127.0.0.1:8000/api/predictions
```
Zwraca aktualne predykcje zapisane w pliku CSV.

Najbardziej podejrzane predykcje
GET /api/predictions/top?limit=20

Przykład:
```
http://127.0.0.1:8000/api/predictions/top?limit=20
```
Zwraca loty z najwyższym prawdopodobieństwem anomalii.

10. Uruchomienie przez Docker

Jeżeli projekt posiada plik Dockerfile, można zbudować obraz:

docker build -t gnss-adsb-spoofing-dashboard .

Następnie uruchomić kontener:
```
docker run --rm -p 8000:8000 gnss-adsb-spoofing-dashboard
```
Aplikacja będzie dostępna pod adresem:
```
http://127.0.0.1:8000
```
W czasie developmentu można uruchomić kontener z podpiętym katalogiem projektu:
```
docker run --rm -p 8000:8000 -v "${PWD}:/app" gnss-adsb-spoofing-dashboard
```
11. Typowa kolejność uruchomienia demo

Najprostszy workflow demonstracyjny:
```
pip install -r requirements.txt
python model_pipeline.py
python train_model_grouped_v2.py
python scripts/generate_models_dashboard.py
python scripts/prepare_av_gps_dataset.py
uvicorn app:app --host 127.0.0.1 --port 8000
```
Opcjonalnie, w drugim terminalu można uruchomić worker OpenSky:

python opensky_worker.py
12. Uwagi dotyczące OpenSky API

OpenSky Network API posiada limity zapytań. W przypadku przekroczenia limitu worker może otrzymać odpowiedź:
```
429 Too Many Requests
```
W takiej sytuacji worker odczekuje określony czas przed kolejną próbą pobrania danych. Na czas pracy nad dashboardem lub prezentacją można wyłączyć automatyczne pobieranie danych live, ustawiając w app.py:

ENABLE_OPENSKY_WORKER = False

Aplikacja może wtedy korzystać z wcześniej wygenerowanych plików:
```
data/opensky_latest_predictions.csv
static/generated/opensky_live_map.html
```
## Ograniczenia projektu


Model ADS-B wykrywa anomalie na podstawie trajektorii i parametrów wiadomości, ale nie analizuje bezpośrednio sygnału radiowego.
Model GNSS-SDR działa na cechach sygnałowych, ale nie analizuje pełnej trajektorii lotu.
Dane live z OpenSky mogą być ograniczone przez dostępność API i rate limity.
Wyniki modeli zależą od jakości danych treningowych oraz sposobu podziału zbioru treningowego i testowego.
Wykrycie anomalii nie oznacza automatycznie potwierdzonego ataku — może wskazywać również na błąd danych, brak pozycji, opóźnienie kontaktu lub nietypowe zachowanie obiektu.
Możliwe kierunki rozwoju
Dodanie modelu łączącego dane ADS-B i GNSS-SDR.
Dodanie obsługi wielu źródeł API lotniczych.
Rozbudowa mapy live o pełne trajektorie i historię lotów.
Dodanie alertów dla wysokiego prawdopodobieństwa spoofingu.
Dodanie panelu porównującego więcej algorytmów ML.
Wykorzystanie modeli sekwencyjnych, np. LSTM lub Transformerów.
Rozbudowa detekcji o reguły fizyczne, np. maksymalna dopuszczalna zmiana prędkości lub kierunku.
Eksport raportów do HTML/PDF.

## Autorzy

- Paweł Wypych
- Antoni Wałach

Projekt przygotowany w ramach Hackathonu Kościuszkon - detekcja anomalii i spoofingu w systemach nawigacyjnych GNSS oraz danych ADS-B.

dane pobrane z:
https://data.mendeley.com/datasets/6fhw732ccz/1
https://zenodo.org/records/17413258
https://github.com/gnss-sdr/gnss-sdr?utm_source=chatgpt.com
