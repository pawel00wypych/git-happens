import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv("features/all_gnss_features.csv")

feature_cols = [
    "n_epochs",
    "cn0_original_mean",
    "cn0_original_std",
    "prompt_abs_mean",
    "prompt_abs_std",
    "doppler_mean",
    "doppler_std",
    "doppler_range",
    "carrier_error_std",
    "carrier_error_abs_mean",
    "carrier_phase_std",
    "code_error_std",
    "code_error_abs_mean",
    "code_error_filt_abs_mean",
    "code_freq_mean",
    "code_freq_std",
    "early_late_balance_mean",
    "early_late_balance_std",
    "early_late_asymmetry_mean",
    "early_late_asymmetry_std",
    "prompt_dominance_mean",
    "prompt_dominance_std",
]

X = df[feature_cols]
y = df["label"]

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
)

scores = cross_val_score(model, X, y, cv=3)

print("CV scores:", scores)
print("Mean:", scores.mean())