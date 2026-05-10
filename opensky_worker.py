import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
import joblib
import folium

class OpenSkyRateLimitError(Exception):
    def __init__(self, retry_after_seconds: int):
        super().__init__(f"OpenSky rate limit exceeded. Retry after {retry_after_seconds} seconds.")
        self.retry_after_seconds = retry_after_seconds

# ============================================================
# CONFIG
# ============================================================

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "adsb_attack_detector_trajectory.pkl"
FEATURES_FILE = BASE_DIR / "models" / "model_features_trajectory.pkl"


DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = BASE_DIR / "static" / "generated"

DATA_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV = DATA_DIR / "opensky_history.csv"
LATEST_PREDICTIONS_CSV = DATA_DIR / "opensky_latest_predictions.csv"
LIVE_MAP_HTML = GENERATED_DIR / "opensky_live_map.html"

# Niższy threshold = większa czułość na anomalie.
FINAL_THRESHOLD = 0.3
# Loty pokazywane w panelu wysokiego ryzyka.
HIGH_RISK_THRESHOLD = 0.6

# OpenSky może mieć rate limit. 15 sekund działa do demo,
# ale jeśli dostaniesz 429, zwiększ na 30 albo 60.
POLL_INTERVAL_SECONDS = 60
MAX_POSITION_AGE_SECONDS = 60

# Ile ostatnich snapshotów trzymać w historii.
MAX_HISTORY_ROWS = 50_000

# Ile ostatnich punktów trajektorii pokazywać na mapie dla jednego samolotu.
TRAJECTORY_HISTORY_POINTS = 30

# Bounding box: Polska / Europa Środkowa.
BOUNDING_BOX = {
    "lamin": 48.5,
    "lomin": 13.5,
    "lamax": 59.5,
    "lomax": 27.0,
}

# Opcjonalnie: jeśli chcesz testować jeden samolot, wpisz ICAO24.
# Jeśli None, pobiera samoloty z bounding boxa.
ICAO24_FILTER = None
# ICAO24_FILTER = "0c20b6"


# ============================================================
# OPENSKY CLIENT
# ============================================================

def fetch_opensky_states() -> pd.DataFrame:
    params = {
        "extended": 1
    }

    if ICAO24_FILTER:
        params["icao24"] = ICAO24_FILTER
    else:
        params.update(BOUNDING_BOX)

    response = requests.get(
        OPENSKY_STATES_URL,
        params=params,
        timeout=30,
    )

    if response.status_code == 429:
        retry_after = (
                response.headers.get("X-Rate-Limit-Retry-After-Seconds")
                or response.headers.get("Retry-After")
                or "600"
        )
        raise OpenSkyRateLimitError(int(float(retry_after)))

    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenSky API error {response.status_code}: {response.text}"
        )

    data = response.json()

    states = data.get("states") or []
    api_time = data.get("time")

    columns = [
        "icao24",
        "callsign",
        "origin_country",
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "baro_altitude",
        "on_ground",
        "velocity",
        "true_track",
        "vertical_rate",
        "sensors",
        "geo_altitude",
        "squawk",
        "spi",
        "position_source",
        "category",
    ]

    if not states:
        return pd.DataFrame(columns=columns + ["api_time"])

    fixed_states = []

    for state in states:
        state = list(state)

        if len(state) < len(columns):
            state = state + [None] * (len(columns) - len(state))
        elif len(state) > len(columns):
            state = state[:len(columns)]

        fixed_states.append(state)

    df = pd.DataFrame(fixed_states, columns=columns)
    df["api_time"] = api_time

    return df


def add_high_risk_panel_to_map(
    m: folium.Map,
    latest_map_df: pd.DataFrame,
    threshold: float = 0.6,
):
    """
    Dodaje do mapy otwierany/zamykany panel z lotami,
    dla których attack_probability > threshold.
    """

    if latest_map_df.empty or "attack_probability" not in latest_map_df.columns:
        high_risk_df = pd.DataFrame()
    else:
        high_risk_df = latest_map_df.copy()
        high_risk_df["attack_probability"] = pd.to_numeric(
            high_risk_df["attack_probability"],
            errors="coerce"
        )

        high_risk_df = high_risk_df[
            high_risk_df["attack_probability"] > threshold
        ].copy()

        high_risk_df = high_risk_df.sort_values(
            "attack_probability",
            ascending=False
        )

    rows_html = ""

    if high_risk_df.empty:
        rows_html = """
        <tr>
            <td colspan="7" style="padding: 8px; text-align: center; color: #666;">
                Brak lotów z wysokim prawdopodobieństwem anomalii.
            </td>
        </tr>
        """
    else:
        for _, row in high_risk_df.iterrows():
            icao24 = row.get("icao24", "")
            callsign = row.get("callsign", "")
            country = row.get("origin_country", "")
            probability = row.get("attack_probability", 0)
            velocity = row.get("velocity", "")
            heading = row.get("heading", "")
            altitude = row.get("geoaltitude", "")

            if pd.isna(callsign):
                callsign = ""

            rows_html += f"""
            <tr>
                <td>{callsign}</td>
                <td>{icao24}</td>
                <td>{country}</td>
                <td><b>{float(probability):.3f}</b></td>
                <td>{velocity}</td>
                <td>{heading}</td>
                <td>{altitude}</td>
            </tr>
            """

    high_risk_count = len(high_risk_df)

    panel_html = f"""
    <div id="high-risk-toggle" style="
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
    ">
        <button onclick="toggleHighRiskPanel()" style="
            background: #b00020;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        ">
            High risk flights ({high_risk_count})
        </button>
    </div>

    <div id="high-risk-panel" style="
        display: none;
        position: fixed;
        top: 70px;
        right: 20px;
        width: 760px;
        max-height: 520px;
        overflow-y: auto;
        background: white;
        border: 2px solid #b00020;
        border-radius: 10px;
        z-index: 10000;
        font-size: 13px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    ">
        <div style="
            position: sticky;
            top: 0;
            background: #b00020;
            color: white;
            padding: 10px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <span>Flights with attack probability &gt; {threshold}</span>
            <button onclick="toggleHighRiskPanel()" style="
                background: white;
                color: #b00020;
                border: none;
                border-radius: 5px;
                padding: 4px 8px;
                cursor: pointer;
                font-weight: bold;
            ">
                X
            </button>
        </div>

        <div style="padding: 10px;">
            <table style="
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            ">
                <thead>
                    <tr style="background: #f3f3f3;">
                        <th style="border: 1px solid #ddd; padding: 6px;">Callsign</th>
                        <th style="border: 1px solid #ddd; padding: 6px;">ICAO24</th>
                        <th style="border: 1px solid #ddd; padding: 6px;">Country</th>
                        <th style="border: 1px solid #ddd; padding: 6px;">P(anomaly)</th>
                        <th style="border: 1px solid #ddd; padding: 6px;">Velocity</th>
                        <th style="border: 1px solid #ddd; padding: 6px;">Heading</th>
                        <th style="border: 1px solid #ddd; padding: 6px;">Geo alt.</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function toggleHighRiskPanel() {{
            var panel = document.getElementById("high-risk-panel");
            if (panel.style.display === "none" || panel.style.display === "") {{
                panel.style.display = "block";
            }} else {{
                panel.style.display = "none";
            }}
        }}
    </script>
    """

    m.get_root().html.add_child(folium.Element(panel_html))

# ============================================================
# NORMALIZATION: OpenSky -> ADS-B training format
# ============================================================

def normalize_opensky_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    normalized = pd.DataFrame()

    normalized["snapshot_time"] = df["api_time"]
    normalized["time"] = df["api_time"]

    normalized["icao24"] = df["icao24"]
    normalized["lat"] = df["latitude"]
    normalized["lon"] = df["longitude"]
    normalized["velocity"] = df["velocity"]
    normalized["heading"] = df["true_track"]
    normalized["vertrate"] = df["vertical_rate"]
    normalized["callsign"] = df["callsign"].fillna("").astype(str).str.strip()
    normalized["onground"] = df["on_ground"]
    normalized["spi"] = df["spi"]
    normalized["squawk"] = df["squawk"]
    normalized["baroaltitude"] = df["baro_altitude"]
    normalized["geoaltitude"] = df["geo_altitude"]

    normalized["lastposupdate"] = df["time_position"]
    normalized["lastcontact"] = df["last_contact"]

    # Jeśli time_position jest puste, zostawiamy NaN.
    # Nie podstawiamy api_time, bo wtedy stara/nieznana pozycja wyglądałaby jak świeża.
    normalized["lastposupdate"] = pd.to_numeric(normalized["lastposupdate"], errors="coerce")
    normalized["lastcontact"] = pd.to_numeric(normalized["lastcontact"], errors="coerce")
    normalized["snapshot_time"] = pd.to_numeric(normalized["snapshot_time"], errors="coerce")

    normalized["position_age_seconds"] = normalized["snapshot_time"] - normalized["lastposupdate"]
    normalized["contact_age_seconds"] = normalized["snapshot_time"] - normalized["lastcontact"]

    # Dodatkowe pola do mapy/CSV
    normalized["origin_country"] = df["origin_country"]
    normalized["position_source"] = df["position_source"]
    normalized["category"] = df["category"]
    normalized["snapshot_utc"] = datetime.now(timezone.utc).isoformat()

    return normalized


# ============================================================
# HISTORY
# ============================================================

def append_to_history(new_df: pd.DataFrame) -> pd.DataFrame:
    if HISTORY_CSV.exists():
        history_df = pd.read_csv(HISTORY_CSV)
        combined = pd.concat([history_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    # Usuwamy duplikaty tego samego samolotu w tym samym czasie.
    combined = combined.drop_duplicates(
        subset=["icao24", "lastposupdate", "lat", "lon"],
        keep="last"
    )

    # Trzymamy ograniczoną historię.
    if len(combined) > MAX_HISTORY_ROWS:
        combined = combined.tail(MAX_HISTORY_ROWS)

    combined.to_csv(HISTORY_CSV, index=False)

    return combined


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_adsb_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["onground", "spi"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map({
                "TRUE": 1,
                "FALSE": 0,
                "1": 1,
                "0": 0,
                "NAN": 0,
                "NONE": 0,
            })

    numeric_candidates = [
        "time",
        "lat",
        "lon",
        "velocity",
        "heading",
        "vertrate",
        "squawk",
        "baroaltitude",
        "geoaltitude",
        "lastposupdate",
        "lastcontact",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = [
        "time",
        "icao24",
        "lat",
        "lon",
        "velocity",
        "heading",
        "vertrate",
        "onground",
        "spi",
        "squawk",
        "baroaltitude",
        "geoaltitude",
        "lastposupdate",
        "lastcontact",
    ]

    for col in required_cols:
        if col not in df.columns:
            if col == "icao24":
                df[col] = "unknown"
            else:
                df[col] = 0

    df = df.sort_values(["icao24", "time"]).copy()

    group = df.groupby("icao24")

    df["delta_time"] = group["time"].diff()
    df["delta_lat"] = group["lat"].diff()
    df["delta_lon"] = group["lon"].diff()
    df["delta_velocity"] = group["velocity"].diff()
    df["delta_heading"] = group["heading"].diff()
    df["delta_baroaltitude"] = group["baroaltitude"].diff()
    df["delta_geoaltitude"] = group["geoaltitude"].diff()

    df["altitude_diff"] = df["geoaltitude"] - df["baroaltitude"]
    df["contact_delay"] = df["lastcontact"] - df["lastposupdate"]

    df["position_change"] = np.sqrt(
        df["delta_lat"] ** 2 + df["delta_lon"] ** 2
    )

    df["velocity_change_rate"] = df["delta_velocity"] / df["delta_time"].replace(0, np.nan)
    df["heading_change_rate"] = df["delta_heading"] / df["delta_time"].replace(0, np.nan)
    df["altitude_change_rate"] = df["delta_geoaltitude"] / df["delta_time"].replace(0, np.nan)

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    return df


def build_model_input(df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    X = df.copy()

    drop_cols = [
        "label",
        "binary_label",
        "icao24",
        "callsign",
        "lat",
        "lon",
        "time",
        "lastposupdate",
        "lastcontact",
        "rss",
        "doppler",
        "origin_country",
        "snapshot_utc",
    ]

    X = X.drop(columns=[col for col in drop_cols if col in X.columns])

    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns
    if len(non_numeric_cols) > 0:
        X = X.drop(columns=non_numeric_cols)

    for col in model_features:
        if col not in X.columns:
            X[col] = 0

    X = X[model_features]

    return X


# ============================================================
# MAP COLORS
# ============================================================

def color_for_icao24(icao24: str) -> str:
    """
    Stały kolor dla każdego samolotu na podstawie ICAO24.
    Ten sam ICAO24 zawsze dostaje ten sam kolor.
    """
    palette = [
        "blue",
        "purple",
        "orange",
        "darkred",
        "cadetblue",
        "darkgreen",
        "darkblue",
        "pink",
        "lightblue",
        "black",
        "gray",
        "darkpurple",
        "beige",
        "lightgreen",
        "red",
        "lightred",
    ]

    if pd.isna(icao24):
        return "gray"

    key = str(icao24).strip().lower()

    if not key:
        return "gray"

    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(palette)

    return palette[idx]


# ============================================================
# MAP
# ============================================================

def add_trajectory_points_to_map(
    m: folium.Map,
    points: list,
    color: str = "gray",
    icao24: str = "",
    callsign: str = "",
):
    """
    Dodaje trajektorię jako punkty, bez linii.

    points: lista [[lat, lon], [lat, lon], ...]
    """

    for idx, (lat, lon) in enumerate(points):
        is_last = idx == len(points) - 1

        radius = 5 if is_last else 3
        opacity = 0.9 if is_last else 0.6

        popup = f"""
        <b>ICAO24:</b> {icao24}<br>
        <b>Callsign:</b> {callsign}<br>
        <b>Trajectory point:</b> {idx + 1}/{len(points)}
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            opacity=opacity,
            popup=popup,
        ).add_to(m)


def save_live_map(latest_df: pd.DataFrame, history_df: pd.DataFrame, output_file: Path):
    latest_map_df = latest_df.copy()
    history_map_df = history_df.copy()

    latest_map_df = latest_map_df.dropna(subset=["lat", "lon"])
    latest_map_df = latest_map_df[
        (latest_map_df["lat"] != 0) &
        (latest_map_df["lon"] != 0)
    ]

    if "position_age_seconds" in latest_map_df.columns:
        latest_map_df["position_age_seconds"] = pd.to_numeric(
            latest_map_df["position_age_seconds"],
            errors="coerce"
        )

        latest_map_df = latest_map_df[
            latest_map_df["position_age_seconds"].notna() &
            (latest_map_df["position_age_seconds"] <= MAX_POSITION_AGE_SECONDS)
            ]

    history_map_df = history_map_df.dropna(subset=["lat", "lon"])
    history_map_df = history_map_df[
        (history_map_df["lat"] != 0) &
        (history_map_df["lon"] != 0)
        ]

    if "position_age_seconds" in history_map_df.columns:
        history_map_df["position_age_seconds"] = pd.to_numeric(
            history_map_df["position_age_seconds"],
            errors="coerce"
        )

        history_map_df = history_map_df[
            history_map_df["position_age_seconds"].notna() &
            (history_map_df["position_age_seconds"] <= MAX_POSITION_AGE_SECONDS)
            ]

    if latest_map_df.empty:
        print("No valid latest lat/lon points for map.")
        return

    center_lat = latest_map_df["lat"].mean()
    center_lon = latest_map_df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap",
    )

    # ============================================================
    # DRAW TRAJECTORIES AS POINTS
    # ============================================================

    for icao24, aircraft_history in history_map_df.groupby("icao24"):
        aircraft_history = aircraft_history.sort_values("time").tail(TRAJECTORY_HISTORY_POINTS)

        if len(aircraft_history) < 2:
            continue

        points = aircraft_history[["lat", "lon"]].values.tolist()
        trajectory_color = color_for_icao24(icao24)

        callsigns = aircraft_history["callsign"].dropna().astype(str).str.strip()
        callsign = callsigns.iloc[-1] if len(callsigns) > 0 else ""

        add_trajectory_points_to_map(
            m=m,
            points=points,
            color=trajectory_color,
            icao24=icao24,
            callsign=callsign,
        )

    # ============================================================
    # DRAW LATEST POINTS WITH MODEL PREDICTION
    # ============================================================

    for _, row in latest_map_df.iterrows():
        predicted = int(row["predicted_anomaly"])
        probability = float(row["attack_probability"])

        icao24 = row.get("icao24", "")
        trajectory_color = color_for_icao24(icao24)

        if predicted == 1:
            border_color = "red"
            status = "Predicted anomaly"
            radius = 8
        else:
            border_color = "green"
            status = "Predicted normal"
            radius = 6

        popup = f"""
        <b>Status:</b> {status}<br>
        <b>Attack probability:</b> {probability:.3f}<br>
        <hr>
        <b>ICAO24:</b> {row.get("icao24", "")}<br>
        <b>Callsign:</b> {row.get("callsign", "")}<br>
        <b>Country:</b> {row.get("origin_country", "")}<br>
        <hr>
        <b>Position age:</b> {row.get("position_age_seconds", "")} s<br>
        <b>Contact age:</b> {row.get("contact_age_seconds", "")} s<br>
        <hr>
        <b>Velocity:</b> {row.get("velocity", "")}<br>
        <b>Heading:</b> {row.get("heading", "")}<br>
        <b>Vertical rate:</b> {row.get("vertrate", "")}<br>
        <b>Baro altitude:</b> {row.get("baroaltitude", "")}<br>
        <b>Geo altitude:</b> {row.get("geoaltitude", "")}<br>
        <hr>
        <b>Position change:</b> {row.get("position_change", "")}<br>
        <b>Velocity change rate:</b> {row.get("velocity_change_rate", "")}<br>
        <b>Heading change rate:</b> {row.get("heading_change_rate", "")}<br>
        <b>Altitude change rate:</b> {row.get("altitude_change_rate", "")}<br>
        <hr>
        <b>Trajectory color:</b> {trajectory_color}<br>
        <b>Snapshot:</b> {row.get("snapshot_utc", "")}<br>
        """

        tooltip_callsign = row.get("callsign", "")
        if pd.isna(tooltip_callsign):
            tooltip_callsign = ""

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=border_color,          # obwódka: predykcja
            weight=3,
            fill=True,
            fill_color=trajectory_color, # środek: kolor lotu/trajektorii
            fill_opacity=0.95,
            popup=popup,
            tooltip=f'{tooltip_callsign} | {icao24} | P(anomaly)={probability:.2f}',
        ).add_to(m)


    last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 300px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
    ">
    <b>Live OpenSky ADS-B inference</b><br>
    <span style="color:green;">●</span> Latest point border: predicted normal<br>
    <span style="color:red;">●</span> Latest point border: predicted anomaly<br>
    <span style="color:gray;">●</span> Trajectory points: unique color per aircraft<br>
    <hr>
    Threshold: {FINAL_THRESHOLD}<br>
    Last map file update: {last_update}<br>
    Aircraft shown: {len(latest_map_df)}<br>
    Trajectory points per aircraft: last {TRAJECTORY_HISTORY_POINTS}<br>
    Data polling interval: {POLL_INTERVAL_SECONDS}s<br>
    <hr>
    <button onclick="window.location.reload();" style="
        padding: 8px 12px;
        font-size: 14px;
        cursor: pointer;
        border: 1px solid #777;
        border-radius: 6px;
        background: #f5f5f5;
    ">
    Odśwież widok mapy
    </button>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    add_high_risk_panel_to_map(
        m=m,
        latest_map_df=latest_map_df,
        threshold=HIGH_RISK_THRESHOLD,
    )

    m.save(str(output_file))


# ============================================================
# ONE ITERATION
# ============================================================

def run_once(model, model_features):
    print("=" * 80)
    print("Fetching OpenSky live data...")

    raw_df = fetch_opensky_states()

    if raw_df.empty:
        print("No aircraft returned.")
        return

    print("Raw states:", raw_df.shape)

    normalized_df = normalize_opensky_df(raw_df)
    history_df = append_to_history(normalized_df)

    print("History rows:", history_df.shape)

    features_df = prepare_adsb_features(history_df)

    # Predykcję pokazujemy tylko dla najnowszego snapshotu.
    latest_time = normalized_df["time"].max()
    latest_df = features_df[features_df["time"] == latest_time].copy()

    if latest_df.empty:
        print("No latest rows after feature engineering.")
        return

    X_latest = build_model_input(latest_df, model_features)

    y_proba = model.predict_proba(X_latest)[:, 1]
    y_pred = (y_proba >= FINAL_THRESHOLD).astype(int)

    latest_df["attack_probability"] = y_proba
    latest_df["predicted_anomaly"] = y_pred

    latest_df.to_csv(LATEST_PREDICTIONS_CSV, index=False)

    print("Latest aircraft:", len(latest_df))
    print("Predicted anomaly distribution:")
    print(latest_df["predicted_anomaly"].value_counts())

    print("Top suspicious aircraft:")
    columns_to_show = [
        "icao24",
        "callsign",
        "lat",
        "lon",
        "velocity",
        "heading",
        "baroaltitude",
        "geoaltitude",
        "attack_probability",
        "predicted_anomaly",
    ]

    print(
        latest_df.sort_values("attack_probability", ascending=False)[
            columns_to_show
        ].head(10)
    )

    save_live_map(latest_df, features_df, LIVE_MAP_HTML)

    print("Saved latest predictions:", LATEST_PREDICTIONS_CSV)
    print("Saved live map:", LIVE_MAP_HTML)


# ============================================================
# MAIN LOOP
# ============================================================

def run_opensky_worker(stop_event=None):
    model = joblib.load(MODEL_FILE)
    model_features = joblib.load(FEATURES_FILE)

    print("Loaded model:", MODEL_FILE)
    print("Loaded features:", FEATURES_FILE)
    print("Map will be updated here:", LIVE_MAP_HTML)

    while True:
        if stop_event is not None and stop_event.is_set():
            print("OpenSky worker stopped.")
            break

        try:
            run_once(model, model_features)
            wait_seconds = POLL_INTERVAL_SECONDS

        except OpenSkyRateLimitError as e:
            wait_seconds = e.retry_after_seconds + 10
            print(f"Rate limit hit. Waiting {wait_seconds} seconds...")

        except Exception as e:
            wait_seconds = POLL_INTERVAL_SECONDS
            print("ERROR:", e)

        print(f"Waiting {wait_seconds} seconds...")

        if stop_event is not None:
            stop_event.wait(wait_seconds)
        else:
            time.sleep(wait_seconds)

if __name__ == "__main__":
    run_opensky_worker()