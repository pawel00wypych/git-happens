import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("Dataset.csv")

print(df.head())
print(df.columns)
print(df["label"].value_counts())

print("Before feature engineering:", df.shape)


# =========================
# BASIC CLEANING
# =========================

df = df.copy()

for col in ["onground", "spi"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map({
            "TRUE": 1,
            "FALSE": 0,
            "True": 1,
            "False": 0
        })


# =========================
# FEATURE ENGINEERING
# =========================

# Ważne: tutaj icao24 musi jeszcze istnieć
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

df["position_change"] = np.sqrt(df["delta_lat"] ** 2 + df["delta_lon"] ** 2)

df["velocity_change_rate"] = df["delta_velocity"] / df["delta_time"].replace(0, np.nan)
df["heading_change_rate"] = df["delta_heading"] / df["delta_time"].replace(0, np.nan)
df["altitude_change_rate"] = df["delta_geoaltitude"] / df["delta_time"].replace(0, np.nan)

print("Before filling NaN:", df.shape)
print(df.isna().sum().sort_values(ascending=False).head(20))

df = df.replace([np.inf, -np.inf], np.nan)

numeric_cols = df.select_dtypes(include=["number"]).columns

for col in numeric_cols:
    if col != "label":
        df[col] = df[col].fillna(0)

print("After filling NaN:", df.shape)
print(df.isna().sum().sort_values(ascending=False).head(20))


# =========================
# LABEL
# =========================

df["binary_label"] = (df["label"] != 0).astype(int)


# =========================
# FEATURES / TARGET
# =========================

# Dopiero tutaj usuwamy icao24 i callsign,
# bo nie chcemy ich używać jako cech modelu.
drop_cols = [
    # target / leakage
    "label",
    "binary_label",

    # identyfikatory tekstowe
    "icao24",
    "callsign",

    # opcjonalnie usuwane, żeby model nie uczył się czasu/lokalizacji
    "lat",
    "lon",
    "time",
    "lastposupdate",
    "lastcontact"
]

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df["binary_label"]

print("X shape:", X.shape)
print("y distribution:")
print(y.value_counts())


# =========================
# TRAIN / TEST
# =========================

from sklearn.model_selection import GroupShuffleSplit

groups = df["icao24"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("y train distribution:")
print(y_train.value_counts())
print("y test distribution:")
print(y_test.value_counts())


# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

y_proba = model.predict_proba(X_test)[:, 1]

for threshold in [0.2, 0.3, 0.35, 0.4, 0.5]:
    y_pred = (y_proba >= threshold).astype(int)

    print("=" * 60)
    print("Threshold:", threshold)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print()
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred))

# =========================
# FINAL THRESHOLD
# =========================

final_threshold = 0.3

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= final_threshold).astype(int)

print("=" * 60)
print("FINAL MODEL RESULTS")
print("Final threshold:", final_threshold)
print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))


# =========================
# FEATURE IMPORTANCE
# =========================

importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(importances.head(20))

# =========================
# PREPARE TEST DATA FOR MAP
# =========================
test_df = df.iloc[test_idx].copy()
test_df["y_true"] = y_test.values
test_df["y_pred"] = y_pred
test_df["attack_probability"] = y_proba

test_df.to_csv("predictions_test.csv", index=False)


import joblib

joblib.dump(model, "adsb_attack_detector.pkl")
joblib.dump(list(X.columns), "model_features.pkl")


# =========================
# MAP VISUALIZATION
# =========================

import folium

# Usuwamy rekordy bez sensownej pozycji
map_df = test_df.copy()

map_df = map_df.dropna(subset=["lat", "lon"])

# Jeżeli wcześniej fillna ustawiło brakujące lat/lon jako 0,
# to wyrzucamy punkty 0,0, bo to nie jest realna pozycja samolotu.
map_df = map_df[
    (map_df["lat"] != 0) &
    (map_df["lon"] != 0)
]

print("Map records:", map_df.shape)

center_lat = map_df["lat"].mean()
center_lon = map_df["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=7,
    tiles="OpenStreetMap"
)


def classify_result(row):
    if row["y_true"] == 1 and row["y_pred"] == 1:
        return "Correctly detected attack", "red"
    elif row["y_true"] == 1 and row["y_pred"] == 0:
        return "Missed attack", "orange"
    elif row["y_true"] == 0 and row["y_pred"] == 1:
        return "False alarm", "purple"
    else:
        return "Correct normal", "green"


for _, row in map_df.iterrows():
    status, color = classify_result(row)

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"""
        <b>Status:</b> {status}<br>
        <b>ICAO24:</b> {row.get("icao24", "")}<br>
        <b>Callsign:</b> {row.get("callsign", "")}<br>
        <b>True label:</b> {row.get("label", "")}<br>
        <b>Binary true:</b> {row["y_true"]}<br>
        <b>Predicted:</b> {row["y_pred"]}<br>
        <b>Attack probability:</b> {row["attack_probability"]:.3f}<br>
        <b>Velocity:</b> {row.get("velocity", "")}<br>
        <b>Heading:</b> {row.get("heading", "")}<br>
        <b>RSS:</b> {row.get("rss", "")}<br>
        <b>Doppler:</b> {row.get("doppler", "")}
        """
    ).add_to(m)


# Legenda
legend_html = """
<div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 260px;
    height: 140px;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
">
<b>ADS-B Attack Detection</b><br>
<span style="color:green;">●</span> Correct normal<br>
<span style="color:red;">●</span> Correctly detected attack<br>
<span style="color:orange;">●</span> Missed attack<br>
<span style="color:purple;">●</span> False alarm<br>
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

m.save("adsb_predictions_map.html")

print("Saved adsb_predictions_map.html")