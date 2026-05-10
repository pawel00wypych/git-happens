from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, spectrogram
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BIN_DIR = BASE_DIR / "data" / "radio"
OUTPUT_DIR = BASE_DIR / "static" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "bin_signal_comparison.html"

# Jeśli Twoje próbki były nagrywane z fs = 25 MHz
FS = 25e6

# Ile par IQ czytać z początku pliku.
# 1_000_000 IQ samples = 2_000_000 bajtów przy int8 IQ.
IQ_SAMPLES_TO_READ = 1_000_000

# Do wykresów czasowych bierzemy mniej punktów, żeby HTML nie był gigantyczny.
TIME_PLOT_SAMPLES = 10_000

# Do scattera jeszcze mniej.
SCATTER_SAMPLES = 20_000

# Typ danych. Najczęściej dla surowych IQ z takich datasetów jest int16.
DTYPE = np.int16

FILES = {
    "Clear Sky": "clear_sky_5000mb.bin",
    "Spoof 1": "spoof_1_5000mb.bin",
    "Spoof 2": "spoof_2_5000mb.bin",
    "Spoof 4": "spoof_4_5000mb.bin",
}


# ============================================================
# LOAD IQ
# ============================================================

def load_iq_int8(path: Path, iq_samples: int) -> np.ndarray:
    """
    Czyta interleaved IQ:
    I0, Q0, I1, Q1, ...
    i zwraca complex64: I + jQ.
    """

    count = iq_samples * 2

    raw = np.fromfile(path, dtype=DTYPE, count=count)

    if len(raw) < 2:
        raise ValueError(f"File too small or empty: {path}")

    if len(raw) % 2 != 0:
        raw = raw[:-1]

    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)

    iq = i + 1j * q

    return iq


def signal_stats(name: str, iq: np.ndarray) -> dict:
    amplitude = np.abs(iq)
    power = amplitude ** 2

    return {
        "scenario": name,
        "iq_samples": len(iq),
        "i_mean": float(np.mean(iq.real)),
        "q_mean": float(np.mean(iq.imag)),
        "i_std": float(np.std(iq.real)),
        "q_std": float(np.std(iq.imag)),
        "amplitude_mean": float(np.mean(amplitude)),
        "amplitude_std": float(np.std(amplitude)),
        "power_mean": float(np.mean(power)),
        "power_std": float(np.std(power)),
        "power_max": float(np.max(power)),
    }


def section(title: str, body: str) -> str:
    return f"""
    <section class="section">
        <h2>{title}</h2>
        {body}
    </section>
    """


def fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ============================================================
# PLOTS
# ============================================================

def build_time_domain_plot(signals: dict[str, np.ndarray]) -> str:
    fig = go.Figure()

    for name, iq in signals.items():
        n = min(TIME_PLOT_SAMPLES, len(iq))
        amplitude = np.abs(iq[:n])

        fig.add_trace(go.Scatter(
            x=np.arange(n),
            y=amplitude,
            mode="lines",
            name=name
        ))

    fig.update_layout(
        title="Amplitude preview in time domain",
        xaxis_title="Sample",
        yaxis_title="Amplitude"
    )

    return section("Time-domain amplitude comparison", fig_html(fig))


def build_iq_scatter_sections(signals: dict[str, np.ndarray]) -> list[str]:
    sections = []

    for name, iq in signals.items():
        n = min(SCATTER_SAMPLES, len(iq))

        df = pd.DataFrame({
            "I": iq[:n].real,
            "Q": iq[:n].imag,
        })

        fig = px.scatter(
            df,
            x="I",
            y="Q",
            title=f"I/Q constellation scatter: {name}",
            opacity=0.35
        )

        fig.update_layout(
            xaxis_title="I",
            yaxis_title="Q"
        )

        sections.append(section(f"I/Q scatter: {name}", fig_html(fig)))

    return sections


def build_psd_plot(signals: dict[str, np.ndarray]) -> str:
    fig = go.Figure()

    for name, iq in signals.items():
        freqs, psd = welch(
            iq,
            fs=FS,
            nperseg=4096,
            return_onesided=False,
            scaling="density"
        )

        # Przesuwamy zero frequency na środek
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)

        psd_db = 10 * np.log10(psd + 1e-12)

        fig.add_trace(go.Scatter(
            x=freqs / 1e6,
            y=psd_db,
            mode="lines",
            name=name
        ))

    fig.update_layout(
        title="Power Spectral Density comparison",
        xaxis_title="Frequency [MHz]",
        yaxis_title="PSD [dB/Hz]"
    )

    return section("Power Spectral Density comparison", fig_html(fig))


def build_spectrogram_sections(signals: dict[str, np.ndarray]) -> list[str]:
    sections = []

    for name, iq in signals.items():
        # Spectrogram jest ciężki, więc bierzemy max 300k próbek.
        n = min(300_000, len(iq))
        x = iq[:n]

        freqs, times, sxx = spectrogram(
            x,
            fs=FS,
            nperseg=2048,
            noverlap=1024,
            return_onesided=False,
            scaling="density",
            mode="psd"
        )

        freqs = np.fft.fftshift(freqs)
        sxx = np.fft.fftshift(sxx, axes=0)

        sxx_db = 10 * np.log10(sxx + 1e-12)

        fig = go.Figure(data=go.Heatmap(
            x=times,
            y=freqs / 1e6,
            z=sxx_db,
            colorbar=dict(title="PSD [dB]")
        ))

        fig.update_layout(
            title=f"Spectrogram: {name}",
            xaxis_title="Time [s]",
            yaxis_title="Frequency [MHz]"
        )

        sections.append(section(f"Spectrogram: {name}", fig_html(fig)))

    return sections


def build_stats_section(stats_rows: list[dict]) -> str:
    df = pd.DataFrame(stats_rows)

    return section(
        "BIN signal statistics",
        df.to_html(index=False)
    )


# ============================================================
# HTML
# ============================================================

def build_html(sections: list[str]) -> str:
    return f"""
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>BIN Signal Comparison</title>

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

        .section h2 {{
            margin-top: 0;
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
        }}

        th {{
            background: #f1f5f9;
        }}

        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <div class="intro">
        <h1>BIN Signal Comparison: Clear Sky vs Spoofing</h1>
        <p>
            Porównanie surowych plików IQ BIN: amplituda w czasie, rozkład I/Q,
            gęstość widmowa mocy PSD oraz spektrogramy dla scenariuszy naturalnych i spoofingowych.
        </p>
        <p>
            Uwaga: pliki są bardzo duże, więc dashboard analizuje tylko wybrany fragment każdego pliku.
        </p>
    </div>

    {''.join(sections)}
</body>
</html>
"""


# ============================================================
# MAIN
# ============================================================

def main():
    signals = {}
    stats_rows = []
    sections = []

    for name, filename in FILES.items():
        path = BIN_DIR / filename

        if not path.exists():
            print(f"Missing file: {path}")
            continue

        print(f"Loading {name}: {path}")

        iq = load_iq_int8(path, IQ_SAMPLES_TO_READ)

        signals[name] = iq
        stats_rows.append(signal_stats(name, iq))

        print(f"Loaded {name}: {len(iq)} IQ samples")

    if not signals:
        raise RuntimeError("No BIN files loaded. Check data/radio/ paths.")

    sections.append(build_stats_section(stats_rows))
    sections.append(build_time_domain_plot(signals))
    sections.extend(build_iq_scatter_sections(signals))
    sections.append(build_psd_plot(signals))
    sections.extend(build_spectrogram_sections(signals))

    html = build_html(sections)

    OUTPUT_FILE.write_text(html, encoding="utf-8")

    print("Saved BIN comparison dashboard:", OUTPUT_FILE)


if __name__ == "__main__":
    main()