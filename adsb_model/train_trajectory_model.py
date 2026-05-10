import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

import json
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =========================
# CONFIG
# =========================

INPUT_FILE = "Dataset.csv"

MODEL_FILE = "adsb_attack_detector_trajectory.pkl"
FEATURES_FILE = "model_features_trajectory.pkl"

FINAL_THRESHOLD = 0.3


# =========================
# SHARED PREPROCESSING
# =========================

def prepare_adsb_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Boolean columns
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

    # Numeric conversion
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

    # Required fields fallback
    if "icao24" not in df.columns:
        df["icao24"] = "unknown"

    if "time" not in df.columns:
        df["time"] = 0

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


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)
print(df.head())
print(df.columns)
print("Label distribution:")
print(df["label"].value_counts())


# =========================
# FEATURE ENGINEERING
# =========================

df = prepare_adsb_features(df)

df["binary_label"] = (df["label"] != 0).astype(int)


# =========================
# FEATURES / TARGET
# =========================

drop_cols = [
    # target
    "label",
    "binary_label",

    # identifiers / text
    "icao24",
    "callsign",

    # absolute position/time removed to make it API-compatible and less leaky
    "lat",
    "lon",
    "time",
    "lastposupdate",
    "lastcontact",

    # unavailable in receiver/API data
    "rss",
    "doppler"
]

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df["binary_label"]

# Keep only numeric features
non_numeric_cols = X.select_dtypes(exclude=["number"]).columns
if len(non_numeric_cols) > 0:
    print("Dropping non-numeric columns:")
    print(non_numeric_cols)
    X = X.drop(columns=non_numeric_cols)

print("X shape:", X.shape)
print("Features:")
print(list(X.columns))
print("y distribution:")
print(y.value_counts())


# =========================
# GROUP SPLIT BY AIRCRAFT
# =========================

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
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)


# =========================
# THRESHOLD TESTING
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
# FINAL RESULT
# =========================

y_pred = (y_proba >= FINAL_THRESHOLD).astype(int)

print("=" * 60)
print("FINAL MODEL")
print("Final threshold:", FINAL_THRESHOLD)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# =========================
# FEATURE IMPORTANCE
# =========================

importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("Top features:")
print(importances.head(20))


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, MODEL_FILE)
joblib.dump(list(X.columns), FEATURES_FILE)

print("Saved model:", MODEL_FILE)
print("Saved features:", FEATURES_FILE)


# =========================
# SAVE MODEL RESULTS FOR DASHBOARD
# =========================

RESULTS_DIR = Path("data/model_results/adsb")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

selected_threshold = FINAL_THRESHOLD

accuracy = accuracy_score(y_test, y_pred)
attack_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
attack_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
attack_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

model_summary = {
    "model_name": "Random Forest ADS-B trajectory-only classifier",
    "model_type": "ADS-B trajectory model",
    "task": "Binary ADS-B message injection anomaly detection",
    "selected_threshold": selected_threshold,
    "accuracy": float(accuracy),
    "attack_precision": float(attack_precision),
    "attack_recall": float(attack_recall),
    "attack_f1": float(attack_f1),
    "validation_method": "GroupShuffleSplit by icao24",
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "normal_test_samples": int((y_test == 0).sum()),
    "attack_test_samples": int((y_test == 1).sum()),
    "description": (
        "Model trenowany na danych ADS-B. Wykorzystuje cechy trajektorii, "
        "takie jak prędkość, heading, wysokość, zmiany pozycji oraz tempo zmian parametrów lotu. "
        "Model nie używa lat/lon/time/RSS/Doppler, aby był kompatybilny z receiver messages i API."
    )
}

with open(RESULTS_DIR / "model_summary.json", "w", encoding="utf-8") as f:
    json.dump(model_summary, f, indent=2, ensure_ascii=False)

# Confusion matrix dla finalnego thresholda
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv(
    RESULTS_DIR / "confusion_matrix.csv",
    index=False,
    header=False
)

# Feature importance
importances.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)

# Threshold results
threshold_rows = []

for threshold in [0.2, 0.3, 0.35, 0.4, 0.5]:
    threshold_pred = (y_proba >= threshold).astype(int)

    cm_threshold = confusion_matrix(y_test, threshold_pred)
    tn, fp = cm_threshold[0]
    fn, tp = cm_threshold[1]

    threshold_rows.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, threshold_pred),
        "attack_precision": precision_score(y_test, threshold_pred, pos_label=1, zero_division=0),
        "attack_recall": recall_score(y_test, threshold_pred, pos_label=1, zero_division=0),
        "attack_f1": f1_score(y_test, threshold_pred, pos_label=1, zero_division=0),
        "false_alarms": int(fp),
        "missed_attacks": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    })

threshold_results_df = pd.DataFrame(threshold_rows)
threshold_results_df.to_csv(
    RESULTS_DIR / "threshold_results.csv",
    index=False
)

print("Saved ADS-B model results to:", RESULTS_DIR)