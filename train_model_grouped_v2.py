from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

DATASET_PATH = Path("features/all_gnss_window_features.csv")
MODEL_PATH = Path("features/random_forest_gnss_spoofing_grouped_v2.joblib")
IMPORTANCE_PATH = Path("features/feature_importance_grouped_v2.csv")
REPORT_PATH = Path("features/grouped_evaluation_report_v2.txt")
PREDICTIONS_PATH = Path("features/grouped_holdout_predictions_v2.csv")
DASHBOARD_RESULTS_DIR = Path("data/model_results/gnss_sdr")
DASHBOARD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_MODEL_SUMMARY_PATH = DASHBOARD_RESULTS_DIR / "model_summary.json"
DASHBOARD_CONFUSION_MATRIX_PATH = DASHBOARD_RESULTS_DIR / "confusion_matrix.csv"
DASHBOARD_FEATURE_IMPORTANCE_PATH = DASHBOARD_RESULTS_DIR / "feature_importance.csv"
DASHBOARD_CV_RESULTS_PATH = DASHBOARD_RESULTS_DIR / "cv_results.csv"
DASHBOARD_THRESHOLD_RESULTS_PATH = DASHBOARD_RESULTS_DIR / "threshold_results.csv"
DASHBOARD_PREDICTIONS_PATH = DASHBOARD_RESULTS_DIR / "holdout_predictions.csv"

TARGET_COL = "is_spoofed"
LABEL_COL = "label"
GROUP_COLS = ["source_file", "tracking_file"]

FEATURE_COLS = [
    "n_epochs",

    "cn0_original_mean",
    "cn0_original_std",
    "cn0_original_median",
    "cn0_original_min",
    "cn0_original_max",
    "cn0_est_prompt",

    "prompt_abs_mean",
    "prompt_abs_std",
    "prompt_abs_median",
    "prompt_abs_min",
    "prompt_abs_max",
    "prompt_i_mean",
    "prompt_i_std",
    "prompt_q_mean",
    "prompt_q_std",

    "doppler_mean",
    "doppler_std",
    "doppler_min",
    "doppler_max",
    "doppler_range",

    "pll_lock_ratio",
    "carrier_error_mean",
    "carrier_error_std",
    "carrier_error_abs_mean",
    "carrier_phase_std",
    "carrier_error_filt_mean",
    "carrier_error_filt_std",

    "code_error_mean",
    "code_error_std",
    "code_error_abs_mean",
    "code_error_filt_mean",
    "code_error_filt_std",
    "code_error_filt_abs_mean",
    "code_freq_mean",
    "code_freq_std",

    "early_late_balance_mean",
    "early_late_balance_std",
    "early_late_balance_median",
    "early_late_balance_min",
    "early_late_balance_max",

    "early_late_asymmetry_mean",
    "early_late_asymmetry_std",
    "early_late_asymmetry_median",
    "early_late_asymmetry_p95",
    "early_late_asymmetry_max",

    "prompt_dominance_mean",
    "prompt_dominance_std",
    "prompt_dominance_median",
    "prompt_dominance_p95",
    "prompt_dominance_max",
]


def make_group_id(df: pd.DataFrame) -> pd.Series:
    missing = [col for col in GROUP_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns: {missing}")

    return df[GROUP_COLS].astype(str).agg("__".join, axis=1)


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    max_depth=None,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def safe_roc_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_proba)


def print_and_collect(lines, text=""):
    print(text)
    lines.append(str(text))


def evaluate_split(model, X_train, X_test, y_train, y_test, test_meta, report_lines):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = safe_roc_auc(y_test, y_proba)

    print_and_collect(report_lines, "\n================ HOLDOUT EVALUATION ================")
    print_and_collect(report_lines, f"Holdout accuracy: {acc}")
    print_and_collect(report_lines, f"Holdout precision: {precision}")
    print_and_collect(report_lines, f"Holdout recall: {recall}")
    print_and_collect(report_lines, f"Holdout F1: {f1}")
    print_and_collect(report_lines, f"Holdout ROC AUC: {auc}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print_and_collect(report_lines, "\nConfusion matrix labels=[clean=0, spoofed=1]:")
    print_and_collect(report_lines, cm)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["clean", "spoofed"],
        zero_division=0,
    )
    print_and_collect(report_lines, "\nClassification report:")
    print_and_collect(report_lines, report)

    predictions = test_meta.copy()
    predictions["y_true"] = y_test.values
    predictions["y_pred"] = y_pred
    predictions["spoof_probability"] = y_proba
    predictions["correct"] = predictions["y_true"] == predictions["y_pred"]

    if LABEL_COL in predictions.columns:
        per_label = predictions.groupby(LABEL_COL).agg(
            count=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            avg_spoof_probability=("spoof_probability", "mean"),
        )
        print_and_collect(report_lines, "\nAccuracy per label:")
        print_and_collect(report_lines, per_label)

    per_group = predictions.groupby(GROUP_COLS).agg(
        count=("correct", "size"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        true_class=("y_true", "first"),
        avg_spoof_probability=("spoof_probability", "mean"),
    )
    print_and_collect(report_lines, "\nAccuracy per source/tracking group:")
    print_and_collect(report_lines, per_group)

    mistakes = predictions[~predictions["correct"]].copy()
    if not mistakes.empty:
        mistakes["confidence_wrong"] = np.where(
            mistakes["y_pred"] == 1,
            mistakes["spoof_probability"],
            1.0 - mistakes["spoof_probability"],
        )
        mistakes = mistakes.sort_values("confidence_wrong", ascending=False)
        print_and_collect(report_lines, "\nMost confident mistakes:")

        cols = [
            col
            for col in [
                LABEL_COL,
                "source_file",
                "tracking_file",
                "prn",
                "window_id",
                "y_true",
                "y_pred",
                "spoof_probability",
                "confidence_wrong",
            ]
            if col in mistakes.columns
        ]

        print_and_collect(report_lines, mistakes[cols].head(30))
    else:
        print_and_collect(report_lines, "\nNo mistakes in holdout split.")

    # Standardowe predykcje
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    # Predykcje dla dashboardu
    predictions.to_csv(DASHBOARD_PREDICTIONS_PATH, index=False)

    print_and_collect(report_lines, f"\nSaved holdout predictions: {PREDICTIONS_PATH}")
    print_and_collect(report_lines, f"Saved dashboard holdout predictions: {DASHBOARD_PREDICTIONS_PATH}")

    # Confusion matrix dla dashboardu
    pd.DataFrame(cm).to_csv(
        DASHBOARD_CONFUSION_MATRIX_PATH,
        index=False,
        header=False,
    )

    # Threshold comparison dla dashboardu
    threshold_rows = []

    for threshold in [0.2, 0.3, 0.35, 0.4, 0.5]:
        threshold_pred = (y_proba >= threshold).astype(int)

        cm_threshold = confusion_matrix(y_test, threshold_pred, labels=[0, 1])
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
    threshold_results_df.to_csv(DASHBOARD_THRESHOLD_RESULTS_PATH, index=False)

    # Model summary dla dashboardu
    model_summary = {
        "model_name": "Random Forest GNSS-SDR radio feature classifier",
        "model_type": "GNSS-SDR signal-level model",
        "task": "Binary GNSS spoofing detection",
        "selected_threshold": 0.5,
        "accuracy": float(acc),
        "attack_precision": float(precision),
        "attack_recall": float(recall),
        "attack_f1": float(f1),
        "roc_auc": None if pd.isna(auc) else float(auc),
        "validation_method": "StratifiedGroupKFold grouped by source_file and tracking_file",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "normal_test_samples": int((y_test == 0).sum()),
        "attack_test_samples": int((y_test == 1).sum()),
        "description": (
            "Model trenowany na cechach uzyskanych z GNSS-SDR. "
            "Wykorzystuje cechy jakości i śledzenia sygnału GNSS, takie jak CN0, Doppler, "
            "PLL lock, błędy nośnej, błędy kodu, asymetrię korelatorów oraz dominację prompt. "
            "Model działa na poziomie sygnału radiowego, a nie na poziomie trajektorii ADS-B."
        ),
    }

    with open(DASHBOARD_MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2, ensure_ascii=False)

    print_and_collect(report_lines, f"Saved dashboard model summary: {DASHBOARD_MODEL_SUMMARY_PATH}")
    print_and_collect(report_lines, f"Saved dashboard confusion matrix: {DASHBOARD_CONFUSION_MATRIX_PATH}")
    print_and_collect(report_lines, f"Saved dashboard threshold results: {DASHBOARD_THRESHOLD_RESULTS_PATH}")

    return model


def main():
    report_lines = []

    df = pd.read_csv(DATASET_PATH)

    print_and_collect(report_lines, f"Dataset shape: {df.shape}")
    print_and_collect(report_lines, "\nColumns:")
    print_and_collect(report_lines, df.columns.tolist())

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    available_features = [col for col in FEATURE_COLS if col in df.columns]
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]

    print_and_collect(report_lines, f"\nAvailable features: {len(available_features)}")
    print_and_collect(report_lines, available_features)

    if missing_features:
        print_and_collect(report_lines, "\nMissing features:")
        print_and_collect(report_lines, missing_features)

    X = df[available_features]
    y = df[TARGET_COL].astype(int)
    groups = make_group_id(df)

    print_and_collect(report_lines, "\nTarget distribution:")
    print_and_collect(report_lines, y.value_counts())

    if LABEL_COL in df.columns:
        print_and_collect(report_lines, "\nRows per label:")
        print_and_collect(report_lines, df.groupby([LABEL_COL, TARGET_COL]).size())

    print_and_collect(report_lines, "\nGroup distribution:")
    print_and_collect(report_lines, f"Unique groups: {groups.nunique()}")
    print_and_collect(report_lines, groups.value_counts().head(30))

    group_target = df.assign(_group=groups).groupby("_group")[TARGET_COL].agg(["first", "nunique", "size"])
    mixed_groups = group_target[group_target["nunique"] > 1]
    if not mixed_groups.empty:
        print_and_collect(report_lines, "\nWARNING: Some groups contain both clean and spoofed rows:")
        print_and_collect(report_lines, mixed_groups.head(20))

    # ==========================================================
    # 1. StratifiedGroupKFold CV
    #    This keeps whole groups together and tries to preserve classes.
    # ==========================================================

    n_splits = min(4, groups.nunique())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_rows = []
    print_and_collect(report_lines, f"\nStratifiedGroupKFold CV ({n_splits} folds):")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = make_model()
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        y_proba = fold_model.predict_proba(X_test)[:, 1]

        fold_acc = accuracy_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred, zero_division=0)
        fold_auc = safe_roc_auc(y_test, y_proba)

        row = {
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_groups": groups.iloc[train_idx].nunique(),
            "test_groups": groups.iloc[test_idx].nunique(),
            "test_clean": int((y_test == 0).sum()),
            "test_spoofed": int((y_test == 1).sum()),
            "accuracy": fold_acc,
            "f1": fold_f1,
            "roc_auc": fold_auc,
        }
        cv_rows.append(row)
        print_and_collect(report_lines, row)

    cv_df = pd.DataFrame(cv_rows)
    print_and_collect(report_lines, "\nCV summary:")
    print_and_collect(report_lines, cv_df[["accuracy", "f1", "roc_auc"]].mean(numeric_only=True))

    cv_df.to_csv(DASHBOARD_CV_RESULTS_PATH, index=False)
    print_and_collect(report_lines, f"\nSaved dashboard CV results: {DASHBOARD_CV_RESULTS_PATH}")

    # ==========================================================
    # 2. Holdout split from one StratifiedGroupKFold fold
    #    Choose the fold with both clean and spoofed in test.
    # ==========================================================

    valid_folds = cv_df[(cv_df["test_clean"] > 0) & (cv_df["test_spoofed"] > 0)]
    if valid_folds.empty:
        raise ValueError(
            "Could not create a grouped holdout containing both classes. "
            "You probably need more clean/spoof groups or different grouping columns."
        )

    # Pick fold closest to 25% test size while containing both classes.
    desired_test_size = 0.25 * len(df)
    selected_fold = int(
        valid_folds.assign(size_distance=(valid_folds["test_rows"] - desired_test_size).abs())
        .sort_values(["size_distance", "fold"])
        .iloc[0]["fold"]
    )

    print_and_collect(report_lines, f"\nSelected holdout fold: {selected_fold}")

    selected_train_idx = selected_test_idx = None
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        if fold == selected_fold:
            selected_train_idx = train_idx
            selected_test_idx = test_idx
            break

    X_train, X_test = X.iloc[selected_train_idx], X.iloc[selected_test_idx]
    y_train, y_test = y.iloc[selected_train_idx], y.iloc[selected_test_idx]
    test_meta_cols = [col for col in [LABEL_COL, *GROUP_COLS, "prn", "window_id", "window_start_row", "window_end_row"] if col in df.columns]
    test_meta = df.iloc[selected_test_idx][test_meta_cols]

    print_and_collect(report_lines, "\nGrouped holdout split:")
    print_and_collect(report_lines, f"Train rows: {len(selected_train_idx)}")
    print_and_collect(report_lines, f"Test rows: {len(selected_test_idx)}")
    print_and_collect(report_lines, f"Train groups: {groups.iloc[selected_train_idx].nunique()}")
    print_and_collect(report_lines, f"Test groups: {groups.iloc[selected_test_idx].nunique()}")

    print_and_collect(report_lines, "\nTrain target distribution:")
    print_and_collect(report_lines, y_train.value_counts())

    print_and_collect(report_lines, "\nTest target distribution:")
    print_and_collect(report_lines, y_test.value_counts())

    model = make_model()
    evaluate_split(model, X_train, X_test, y_train, y_test, test_meta, report_lines)

    # ==========================================================
    # 3. Fit final model on all data and save it
    # ==========================================================

    final_model = make_model()
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    classifier = final_model.named_steps["classifier"]
    importance_df = pd.DataFrame(
        {
            "feature": available_features,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    importance_df.to_csv(DASHBOARD_FEATURE_IMPORTANCE_PATH, index=False)

    print_and_collect(report_lines, f"\nSaved grouped v2 model: {MODEL_PATH}")
    print_and_collect(report_lines, f"Saved grouped v2 feature importance: {IMPORTANCE_PATH}")
    print_and_collect(report_lines, f"Saved dashboard feature importance: {DASHBOARD_FEATURE_IMPORTANCE_PATH}")

    print_and_collect(report_lines, "\nTop 20 features:")
    print_and_collect(report_lines, importance_df.head(20))

    REPORT_PATH.write_text("\n".join(map(str, report_lines)), encoding="utf-8")
    print(f"Saved evaluation report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
