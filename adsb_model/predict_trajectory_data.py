import json
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import folium


# =========================
# CONFIG
# =========================

MODEL_FILE = "adsb_attack_detector_trajectory.pkl"
FEATURES_FILE = "model_features_trajectory.pkl"

FINAL_THRESHOLD = 0.3

# Zmień na:
# "receiver_messages.csv"
# albo "spire_flights.json"
INPUT_FILE = "Receiver messages.csv"

OUTPUT_CSV = "trajectory_predictions.csv"
OUTPUT_MAP = "trajectory_predictions_map.html"


# =========================
# NORMALIZATION
# =========================

def parse_timestamp_to_unix(value):
    if value is None:
        return 0

    if isinstance(value, (int, float)):
        return int(value)

    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0


def normalize_spire_flight(flight: dict) -> dict:
    timestamp = parse_timestamp_to_unix(flight.get("timestamp"))

    return {
        "time": timestamp,
        "icao24": flight.get("icao_address"),
        "lat": flight.get("latitude"),
        "lon": flight.get("longitude"),
        "velocity": flight.get("speed"),
        "heading": flight.get("heading"),
        "vertrate": flight.get("vertical_rate"),
        "callsign": flight.get("callsign"),
        "onground": flight.get("on_ground"),
        "spi": False,
        "squawk": flight.get("squawk"),
        "baroaltitude": flight.get("altitude_baro"),
        "geoaltitude": flight.get("altitude_gnss"),
        "lastposupdate": timestamp,
        "lastcontact": timestamp,
    }


def load_input_file(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Obsługa pojedynczego obiektu JSON
        if isinstance(data, dict):
            # jeżeli API zwróci {"flights": [...]}
            if "flights" in data and isinstance(data["flights"], list):
                rows = [normalize_spire_flight(item) for item in data["flights"]]
            else:
                rows = [normalize_spire_flight(data)]

        # Obsługa listy obiektów JSON
        elif isinstance(data, list):
            rows = [normalize_spire_flight(item) for item in data]

        else:
            raise ValueError("Unsupported JSON structure")

        return pd.DataFrame(rows)

    raise ValueError("Unsupported input file format. Use CSV or JSON.")


# =========================
# FEATURE ENGINEERING
# =========================

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
                "NONE": 0
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
        "doppler"
    ]

    X = X.drop(columns=[col for col in drop_cols if col in X.columns])

    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns
    if len(non_numeric_cols) > 0:
        X = X.drop(columns=non_numeric_cols)

    # Dodaj brakujące kolumny, jeżeli API/drugi plik czegoś nie ma
    for col in model_features:
        if col not in X.columns:
            X[col] = 0

    # Usuń nadmiarowe kolumny i ustaw kolejność jak podczas treningu
    X = X[model_features]

    return X


# =========================
# MAP
# =========================

def save_prediction_map(df: pd.DataFrame, output_file: str):
    map_df = df.copy()

    map_df = map_df.dropna(subset=["lat", "lon"])
    map_df = map_df[
        (map_df["lat"] != 0) &
        (map_df["lon"] != 0)
    ]

    if map_df.empty:
        print("No valid lat/lon points for map.")
        return

    m = folium.Map(
        location=[map_df["lat"].mean(), map_df["lon"].mean()],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    for _, row in map_df.iterrows():
        predicted = int(row["predicted_anomaly"])
        probability = float(row["attack_probability"])

        if predicted == 1:
            color = "red"
            status = "Predicted anomaly"
        else:
            color = "green"
            status = "Predicted normal"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4 if predicted == 1 else 3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=f"""
            <b>Status:</b> {status}<br>
            <b>ICAO24:</b> {row.get("icao24", "")}<br>
            <b>Callsign:</b> {row.get("callsign", "")}<br>
            <b>Attack probability:</b> {probability:.3f}<br>
            <b>Velocity:</b> {row.get("velocity", "")}<br>
            <b>Heading:</b> {row.get("heading", "")}<br>
            <b>Baro altitude:</b> {row.get("baroaltitude", "")}<br>
            <b>GNSS altitude:</b> {row.get("geoaltitude", "")}<br>
            <b>Position change:</b> {row.get("position_change", "")}<br>
            """
        ).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 220px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
    ">
    <b>Trajectory anomaly detection</b><br>
    <span style="color:green;">●</span> Predicted normal<br>
    <span style="color:red;">●</span> Predicted anomaly<br>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(output_file)

    print("Saved map:", output_file)


# =========================
# MAIN
# =========================

model = joblib.load(MODEL_FILE)
model_features = joblib.load(FEATURES_FILE)

df = load_input_file(INPUT_FILE)

print("Loaded input:", INPUT_FILE)
print("Input shape:", df.shape)
print(df.head())

df = prepare_adsb_features(df)
X = build_model_input(df, model_features)

print("Prediction X shape:", X.shape)
print("Features used:")
print(list(X.columns))

y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba >= FINAL_THRESHOLD).astype(int)

df["attack_probability"] = y_proba
df["predicted_anomaly"] = y_pred

print("Predicted anomaly distribution:")
print(df["predicted_anomaly"].value_counts())

print("Anomaly rate:")
print(df["predicted_anomaly"].mean())

print("Top suspicious records:")
print(
    df.sort_values("attack_probability", ascending=False)[[
        "time",
        "icao24",
        "callsign",
        "lat",
        "lon",
        "velocity",
        "heading",
        "baroaltitude",
        "geoaltitude",
        "attack_probability",
        "predicted_anomaly"
    ]].head(20)
)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved predictions:", OUTPUT_CSV)

save_prediction_map(df, OUTPUT_MAP)