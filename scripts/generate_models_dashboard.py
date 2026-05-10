from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "data" / "model_results"
OUTPUT_DIR = BASE_DIR / "static" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "model_dashboard.html"


MODEL_CONFIGS = {
    "adsb": {
        "title": "ADS-B trajectory model",
        "subtitle": "Model wykrywający anomalie na poziomie trajektorii i wiadomości ADS-B.",
        "probability_name": "attack_probability",
        "feature_descriptions": {
            "position_change": "Zmiana pozycji samolotu między kolejnymi komunikatami.",
            "velocity": "Prędkość pozioma samolotu.",
            "heading": "Kierunek lotu w stopniach.",
            "delta_velocity": "Zmiana prędkości względem poprzedniego punktu.",
            "delta_heading": "Zmiana kierunku lotu względem poprzedniego punktu.",
            "velocity_change_rate": "Tempo zmiany prędkości.",
            "heading_change_rate": "Tempo zmiany kierunku lotu.",
            "altitude_diff": "Różnica między wysokością GNSS/geometric i barometryczną.",
            "altitude_change_rate": "Tempo zmiany wysokości.",
            "contact_delay": "Różnica między czasem ostatniej pozycji a ostatnim kontaktem.",
        },
    },
    "gnss_sdr": {
        "title": "GNSS-SDR signal-level model",
        "subtitle": "Model wykrywający spoofing na podstawie cech sygnału GNSS wyliczonych z GNSS-SDR.",
        "probability_name": "spoof_probability",
        "feature_descriptions": {
            "prompt_abs_mean": "Średnia amplituda korelatora prompt. Wysoka ważność sugeruje różnice w sile i strukturze sygnału.",
            "prompt_abs_std": "Zmienność amplitudy prompt w oknie czasowym.",
            "prompt_abs_median": "Mediana amplitudy prompt, bardziej odporna na pojedyncze skoki.",
            "prompt_dominance_p95": "95 percentyl dominacji prompt względem early/late.",
            "prompt_dominance_std": "Zmienność dominacji prompt. Pomaga wykrywać niestabilność korelatorów.",
            "prompt_i_std": "Zmienność składowej I korelatora prompt.",
            "prompt_q_std": "Zmienność składowej Q korelatora prompt.",
            "early_late_asymmetry_mean": "Średnia asymetria korelatorów early/late.",
            "early_late_asymmetry_p95": "Wysoki percentyl asymetrii early/late.",
            "doppler_min": "Minimalna wartość Dopplera w oknie.",
            "doppler_max": "Maksymalna wartość Dopplera w oknie.",
            "carrier_phase_std": "Zmienność fazy nośnej.",
            "code_error_abs_mean": "Średni bezwzględny błąd kodu.",
        },
    },
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


def fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def section(title: str, body: str) -> str:
    return f"""
    <section class="section">
        <h2>{title}</h2>
        {body}
    </section>
    """


def metric_cards(metrics: dict) -> str:
    html = ""

    for name, value in metrics.items():
        html += f"""
        <div class="metric-card">
            <span>{name}</span>
            <strong>{value}</strong>
        </div>
        """

    return f"""
    <div class="metrics-grid">
        {html}
    </div>
    """


def model_paths(model_key: str) -> dict:
    model_dir = RESULTS_DIR / model_key

    return {
        "summary": model_dir / "model_summary.json",
        "cm": model_dir / "confusion_matrix.csv",
        "importance": model_dir / "feature_importance.csv",
        "thresholds": model_dir / "threshold_results.csv",
        "cv": model_dir / "cv_results.csv",
        "predictions": model_dir / "holdout_predictions.csv",
    }


def build_overview_section(summaries: dict) -> str:
    rows = []

    for model_key, summary in summaries.items():
        config = MODEL_CONFIGS[model_key]

        rows.append({
            "model": config["title"],
            "type": summary.get("model_type", "-"),
            "accuracy": summary.get("accuracy", None),
            "precision": summary.get("attack_precision", None),
            "recall": summary.get("attack_recall", None),
            "f1": summary.get("attack_f1", None),
            "roc_auc": summary.get("roc_auc", None),
            "threshold": summary.get("selected_threshold", None),
            "test_rows": summary.get("test_rows", None),
        })

    df = pd.DataFrame(rows)

    table = df.to_html(index=False)

    fig = go.Figure()

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if metric in df.columns and df[metric].notna().any():
            fig.add_trace(go.Bar(
                x=df["model"],
                y=df[metric],
                name=metric,
            ))

    fig.update_layout(
        title="Model metrics comparison",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        barmode="group",
    )

    explanation = """
    <div class="note">
        <p>
            ADS-B model działa na poziomie trajektorii lotu i parametrów wiadomości ADS-B.
            GNSS-SDR model działa na poziomie sygnału radiowego oraz cech śledzenia GNSS.
            Te dwa modele są komplementarne: jeden wykrywa anomalie w ruchu, drugi w samym sygnale.
        </p>
    </div>
    """

    return section(
        "Two-layer detection overview",
        explanation + fig_html(fig) + table
    )


def build_summary_section(model_key: str, summary: dict) -> str:
    config = MODEL_CONFIGS[model_key]

    if not summary:
        return section(
            config["title"],
            f"<p>Brak pliku <code>data/model_results/{model_key}/model_summary.json</code>.</p>"
        )

    metrics = {
        "Accuracy": f'{summary.get("accuracy", 0):.4f}',
        "Precision": f'{summary.get("attack_precision", 0):.4f}',
        "Recall": f'{summary.get("attack_recall", 0):.4f}',
        "F1": f'{summary.get("attack_f1", 0):.4f}',
        "ROC AUC": "-" if summary.get("roc_auc") is None else f'{summary.get("roc_auc", 0):.4f}',
        "Threshold": summary.get("selected_threshold", "-"),
        "Train rows": summary.get("train_rows", "-"),
        "Test rows": summary.get("test_rows", "-"),
    }

    body = f"""
    <p class="subtitle">{config["subtitle"]}</p>
    {metric_cards(metrics)}
    <div class="note">
        <strong>Validation:</strong> {summary.get("validation_method", "-")}<br>
        <strong>Description:</strong> {summary.get("description", "-")}
    </div>
    """

    return section(config["title"], body)


def build_confusion_matrix_section(model_key: str, cm_df: pd.DataFrame | None) -> str:
    config = MODEL_CONFIGS[model_key]

    if cm_df is None:
        return section(
            f"{config['title']} - confusion matrix",
            "<p>Brak pliku confusion_matrix.csv.</p>"
        )

    values = cm_df.values

    fig = px.imshow(
        values,
        text_auto=True,
        title=f"{config['title']} confusion matrix",
        labels=dict(x="Predicted", y="True", color="Count"),
    )

    fig.update_xaxes(ticktext=["Clean/Normal", "Attack/Spoofed"], tickvals=[0, 1])
    fig.update_yaxes(ticktext=["Clean/Normal", "Attack/Spoofed"], tickvals=[0, 1])

    explanation = ""

    if values.shape == (2, 2):
        tn, fp = values[0]
        fn, tp = values[1]

        explanation = f"""
        <div class="matrix-explanation">
            <p><strong>TN:</strong> {tn} — poprawnie rozpoznane próbki clean/normal.</p>
            <p><strong>FP:</strong> {fp} — próbki clean/normal błędnie oznaczone jako atak/spoofing.</p>
            <p><strong>FN:</strong> {fn} — ataki/spoofing przeoczone przez model.</p>
            <p><strong>TP:</strong> {tp} — ataki/spoofing poprawnie wykryte.</p>
        </div>
        """

    return section(
        f"{config['title']} - confusion matrix",
        fig_html(fig) + explanation
    )


def build_threshold_section(model_key: str, df: pd.DataFrame | None) -> str:
    config = MODEL_CONFIGS[model_key]

    if df is None:
        return section(
            f"{config['title']} - threshold comparison",
            "<p>Brak pliku threshold_results.csv.</p>"
        )

    fig = go.Figure()

    for col, label in [
        ("accuracy", "Accuracy"),
        ("attack_precision", "Attack precision"),
        ("attack_recall", "Attack recall"),
        ("attack_f1", "Attack F1"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["threshold"],
                y=df[col],
                mode="lines+markers",
                name=label,
            ))

    fig.update_layout(
        title=f"{config['title']} threshold comparison",
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
    )

    explanation = """
    <div class="note">
        <p>
            Niższy threshold zwykle zwiększa recall, czyli wykrywa więcej ataków,
            ale może powodować więcej false positives. Wyższy threshold zmniejsza liczbę fałszywych alarmów,
            ale może zwiększyć liczbę przeoczonych ataków.
        </p>
    </div>
    """

    return section(
        f"{config['title']} - threshold comparison",
        fig_html(fig) + explanation + df.to_html(index=False)
    )


def build_feature_importance_section(model_key: str, df: pd.DataFrame | None) -> str:
    config = MODEL_CONFIGS[model_key]

    if df is None:
        return section(
            f"{config['title']} - feature importance",
            "<p>Brak pliku feature_importance.csv.</p>"
        )

    if not {"feature", "importance"}.issubset(df.columns):
        return section(
            f"{config['title']} - feature importance",
            "<p>Plik feature_importance.csv musi mieć kolumny feature i importance.</p>"
        )

    top_df = df.sort_values("importance", ascending=False).head(20)

    fig = px.bar(
        top_df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"{config['title']} top 20 features",
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Importance",
        yaxis_title="Feature",
    )

    return section(
        f"{config['title']} - feature importance",
        fig_html(fig)
    )


def build_example_features_section(model_key: str, importance_df: pd.DataFrame | None) -> str:
    config = MODEL_CONFIGS[model_key]
    descriptions = config["feature_descriptions"]

    if importance_df is not None and "feature" in importance_df.columns:
        features = importance_df.sort_values("importance", ascending=False)["feature"].head(10).tolist()
    else:
        features = list(descriptions.keys())[:10]

    rows = ""

    for feature in features:
        importance = "-"

        if importance_df is not None and {"feature", "importance"}.issubset(importance_df.columns):
            match = importance_df[importance_df["feature"] == feature]
            if not match.empty:
                importance = f'{float(match.iloc[0]["importance"]):.4f}'

        description = descriptions.get(
            feature,
            "Cecha używana przez model do oceny prawdopodobieństwa anomalii."
        )

        rows += f"""
        <tr>
            <td><code>{feature}</code></td>
            <td>{importance}</td>
            <td>{description}</td>
        </tr>
        """

    table = f"""
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Importance</th>
                <th>Interpretation</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """

    return section(
        f"{config['title']} - example features",
        table
    )


def build_cv_section(model_key: str, cv_df: pd.DataFrame | None) -> str:
    if model_key != "gnss_sdr":
        return ""

    if cv_df is None:
        return ""

    fig = go.Figure()

    for col in ["accuracy", "f1", "roc_auc"]:
        if col in cv_df.columns:
            fig.add_trace(go.Scatter(
                x=cv_df["fold"],
                y=cv_df[col],
                mode="lines+markers",
                name=col,
            ))

    fig.update_layout(
        title="GNSS-SDR StratifiedGroupKFold CV results",
        xaxis_title="Fold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
    )

    return section(
        "GNSS-SDR grouped cross-validation",
        fig_html(fig) + cv_df.to_html(index=False)
    )


def build_predictions_section(model_key: str, predictions_df: pd.DataFrame | None) -> str:
    config = MODEL_CONFIGS[model_key]

    if predictions_df is None:
        return ""

    probability_col = config["probability_name"]

    if probability_col not in predictions_df.columns:
        return ""

    top_df = predictions_df.sort_values(probability_col, ascending=False).head(20)

    fig = px.histogram(
        predictions_df,
        x=probability_col,
        color="y_true" if "y_true" in predictions_df.columns else None,
        nbins=40,
        title=f"{config['title']} probability distribution",
    )

    body = fig_html(fig)
    body += "<h3>Top 20 highest probability samples</h3>"
    body += top_df.to_html(index=False)

    return section(
        f"{config['title']} - prediction probabilities",
        body
    )


def build_html(sections: list[str]) -> str:
    return f"""
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Models Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

    <style>
        body {{
            margin: 0;
            padding: 24px;
            font-family: Arial, sans-serif;
            background: #f8fafc;
            color: #172033;
        }}

        .intro {{
            background: white;
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }}

        .section {{
            background: white;
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            overflow-x: auto;
        }}

        .subtitle {{
            color: #475569;
            font-size: 15px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(160px, 1fr));
            gap: 14px;
            margin: 18px 0;
        }}

        .metric-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 14px;
        }}

        .metric-card span {{
            display: block;
            color: #64748b;
            font-size: 13px;
        }}

        .metric-card strong {{
            display: block;
            margin-top: 6px;
            font-size: 20px;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
        }}

        th, td {{
            border: 1px solid #e2e8f0;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }}

        th {{
            background: #f1f5f9;
        }}

        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 6px;
        }}

        .note {{
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 12px 16px;
            border-radius: 10px;
            margin: 16px 0;
            color: #334155;
        }}

        .matrix-explanation {{
            background: #f8fafc;
            padding: 14px 16px;
            border-radius: 12px;
            margin-top: 14px;
        }}

        .matrix-explanation p {{
            margin: 6px 0;
        }}
    </style>
</head>
<body>
    <div class="intro">
        <h1>Two-model ADS-B / GNSS Spoofing Detection Dashboard</h1>
        <p>
            Dashboard porównuje dwa modele: model ADS-B działający na poziomie trajektorii
            oraz model GNSS-SDR działający na poziomie cech sygnału radiowego.
        </p>
    </div>

    {''.join(sections)}
</body>
</html>
"""


def main():
    summaries = {}
    loaded = {}

    for model_key in MODEL_CONFIGS.keys():
        paths = model_paths(model_key)

        summary = read_json(paths["summary"])
        cm = read_csv(paths["cm"], header=None)
        importance = read_csv(paths["importance"])
        thresholds = read_csv(paths["thresholds"])
        cv = read_csv(paths["cv"])
        predictions = read_csv(paths["predictions"])

        summaries[model_key] = summary

        loaded[model_key] = {
            "summary": summary,
            "cm": cm,
            "importance": importance,
            "thresholds": thresholds,
            "cv": cv,
            "predictions": predictions,
        }

    sections = []

    sections.append(build_overview_section(summaries))

    for model_key in MODEL_CONFIGS.keys():
        data = loaded[model_key]

        sections.append(build_summary_section(model_key, data["summary"]))
        sections.append(build_confusion_matrix_section(model_key, data["cm"]))
        sections.append(build_threshold_section(model_key, data["thresholds"]))
        sections.append(build_feature_importance_section(model_key, data["importance"]))
        sections.append(build_example_features_section(model_key, data["importance"]))
        cv_section = build_cv_section(model_key, data["cv"])
        if cv_section:
            sections.append(cv_section)

        pred_section = build_predictions_section(model_key, data["predictions"])
        if pred_section:
            sections.append(pred_section)

    html = build_html(sections)

    OUTPUT_FILE.write_text(html, encoding="utf-8")

    print("Saved dashboard:", OUTPUT_FILE)


if __name__ == "__main__":
    main()