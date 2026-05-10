import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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

importances.to_csv("trajectory_feature_importance.csv", index=False)


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, MODEL_FILE)
joblib.dump(list(X.columns), FEATURES_FILE)

print("Saved model:", MODEL_FILE)
print("Saved features:", FEATURES_FILE)