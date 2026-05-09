import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path.cwd()

GPS_CONF = PROJECT_DIR / "gps.conf"

DATASETS = {
    "clear_sky": {
        "display_name": "Clear Sky",
        "file": PROJECT_DIR / "clear_sky_500mb.bin",
    },
    "spoof_1": {
        "display_name": "Spoof 1",
        "file": PROJECT_DIR / "spoof_1_500mb.bin",
    },
    "spoof_2": {
        "display_name": "Spoof 2",
        "file": PROJECT_DIR / "spoof_2_500mb.bin",
    },
    "spoof_4": {
        "display_name": "Spoof 4",
        "file": PROJECT_DIR / "spoof_4_500mb.bin",
    },
}

OUTPUT_ROOT = PROJECT_DIR / "outputs"
FEATURES_ROOT = PROJECT_DIR / "features"

DOCKER_IMAGE = "gnss-sdr"

# Minimum rows required before a tracking file/channel is considered valid.
MIN_TRACKING_ROWS = 100

# Windowing configuration.
# If one tracking row is approximately 1 ms, then 500 rows ~= 0.5 s.
WINDOW_SIZE = 500
WINDOW_STEP = 250          # 50% overlap. Use 500 for no overlap.
MIN_WINDOW_ROWS = 300

# If True, old outputs for each dataset are removed before running GNSS-SDR.
CLEAN_OUTPUT_BEFORE_RUN = True

# If True, print detailed .mat HDF5 structure. This can be noisy.
INSPECT_MAT_FILES = False


# ============================================================
# UTILS
# ============================================================

def ensure_exists(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def to_binary_label(label: str):
    if label == "clear_sky":
        return "clear_sky", 0
    return "spoofed", 1


def safe_mean(g, col):
    return g[col].mean() if col in g.columns else np.nan


def safe_std(g, col):
    return g[col].std() if col in g.columns else np.nan


def safe_min(g, col):
    return g[col].min() if col in g.columns else np.nan


def safe_max(g, col):
    return g[col].max() if col in g.columns else np.nan


def safe_abs_mean(g, col):
    return g[col].abs().mean() if col in g.columns else np.nan


# ============================================================
# 1. RUN GNSS-SDR IN DOCKER
# ============================================================

def run_gnss_sdr(label: str, bin_path: Path, output_dir: Path):
    """
    Runs GNSS-SDR in Docker, using separate output folders per dataset.

    Equivalent to:
    docker run --rm -v "${PWD}:/data" -v "${PWD}/output:/output" \
      gnss-sdr gnss-sdr -c /data/gps.conf -s /data/file.bin
    """

    ensure_exists(GPS_CONF, "gps.conf")
    ensure_exists(bin_path, "Input BIN file")

    output_dir.mkdir(parents=True, exist_ok=True)

    if CLEAN_OUTPUT_BEFORE_RUN:
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    relative_bin = bin_path.relative_to(PROJECT_DIR)
    container_bin_path = f"/data/{relative_bin.as_posix()}"

    command = [
        "docker", "run", "--rm",
        "-v", f"{PROJECT_DIR}:/data",
        "-v", f"{output_dir}:/output",
        DOCKER_IMAGE,
        "gnss-sdr",
        "-c", "/data/gps.conf",
        "-s", container_bin_path,
    ]

    print("\n" + "=" * 80)
    print(f"RUNNING GNSS-SDR: {label}")
    print("=" * 80)
    print("Command:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
    )

    print("\n--- STDOUT ---")
    print(result.stdout)

    print("\n--- STDERR ---")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"GNSS-SDR failed for {label} with exit code {result.returncode}")

    print(f"GNSS-SDR finished for {label}")


# ============================================================
# 2. PRINT HDF5/MAT STRUCTURE
# ============================================================

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET: {name}")
        print(f"  shape: {obj.shape}")
        print(f"  dtype: {obj.dtype}")

        try:
            sample = obj[()]
            if hasattr(sample, "flatten"):
                print(f"  sample: {sample.flatten()[:5]}")
            else:
                print(f"  sample: {sample}")
        except Exception as e:
            print(f"  sample read error: {e}")

    elif isinstance(obj, h5py.Group):
        print(f"GROUP: {name}")


def inspect_mat_files(output_dir: Path):
    print("\n" + "=" * 80)
    print(f"INSPECTING .MAT FILES IN: {output_dir}")
    print("=" * 80)

    mat_files = sorted(output_dir.glob("*.mat"))

    if not mat_files:
        print("No .mat files found.")
        return

    for path in mat_files:
        print("\n" + "=" * 60)
        print(f"FILE: {path.name}")
        print("=" * 60)

        try:
            with h5py.File(path, "r") as f:
                f.visititems(print_hdf5_structure)
        except OSError as e:
            print(f"Cannot read {path.name} as HDF5/MAT v7.3: {e}")


# ============================================================
# 3. CONVERT MAT TO CSV
# ============================================================

def collect_datasets(h5file):
    datasets = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                data = obj[()]
                data = np.array(data)

                if data.size > 0 and np.issubdtype(data.dtype, np.number):
                    datasets[name.replace("/", "_")] = data.flatten()
            except Exception as e:
                print(f"Skipping {name}: {e}")

    h5file.visititems(visitor)
    return datasets


def convert_mat_to_csv(output_dir: Path, csv_dir: Path):
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("CONVERTING .MAT TO .CSV")
    print(f"Input:  {output_dir}")
    print(f"Output: {csv_dir}")
    print("=" * 80)

    mat_files = sorted(output_dir.glob("*.mat"))

    if not mat_files:
        print("No .mat files found.")
        return

    for path in mat_files:
        base_name = path.stem
        output_csv = csv_dir / f"{base_name}.csv"

        print(f"\nReading {path.name}...")

        try:
            with h5py.File(path, "r") as f:
                datasets = collect_datasets(f)

            if not datasets:
                print("  No numeric datasets found.")
                continue

            max_len = max(len(v) for v in datasets.values())
            normalized = {}

            for key, values in datasets.items():
                arr = np.asarray(values).flatten()

                if len(arr) < max_len:
                    padded = np.full(max_len, np.nan)
                    padded[:len(arr)] = arr
                    normalized[key] = padded
                else:
                    normalized[key] = arr[:max_len]

            df = pd.DataFrame(normalized)
            df.to_csv(output_csv, index=False)

            print(f"  Saved: {output_csv}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")

        except OSError as e:
            print(f"  Cannot read {path.name}: {e}")


# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

def estimate_cn0_from_prompt(prompt_values, tint=1e-3):
    """
    Fallback CN0 estimator from Prompt_I + Prompt_Q.
    Use original CN0_SNV_dB_Hz from GNSS-SDR if available.
    """

    prompt_values = np.asarray(prompt_values)

    if len(prompt_values) < 5:
        return np.nan

    p2 = np.abs(prompt_values) ** 2
    p4 = np.abs(prompt_values) ** 4

    M2 = np.mean(p2)
    M4 = np.mean(p4)

    disc = 2 * M2**2 - M4

    if disc <= 0:
        return np.nan

    sqrt_term = np.sqrt(disc)

    noise = M2 - sqrt_term
    carrier = sqrt_term

    if noise <= 0:
        return np.nan

    snr = carrier / noise

    return 10 * np.log10(snr) - 10 * np.log10(tint)


def clean_tracking_df(df):
    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = [
        "PRN",
        "PRN_start_sample_count",
        "Prompt_I",
        "Prompt_Q",
        "abs_E",
        "abs_L",
        "abs_P",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"  Missing required columns: {missing}")
        return pd.DataFrame()

    df = df.dropna(subset=required_cols)

    invalid = (
        (df["PRN"] <= 0) |
        (df["PRN_start_sample_count"] <= 1) |
        (df["abs_P"] <= 1) |
        ((df["Prompt_I"].abs() <= 1) & (df["Prompt_Q"].abs() <= 1))
    )

    df = df[~invalid].copy()

    return df


def build_feature_row(g, label: str, source_file: str, tracking_file: str, prn, window_id, window_start, window_end):
    binary_label, is_spoofed = to_binary_label(label)

    E = g["abs_E"]
    L = g["abs_L"]
    P = g["abs_P"]

    early_late_balance = (E - L) / (E + L + 1e-12)
    early_late_asymmetry = (E - L).abs() / (P + 1e-12)
    prompt_dominance = P / (((E + L) / 2) + 1e-12)

    prompt_complex = g["Prompt_I"].values + 1j * g["Prompt_Q"].values
    cn0_est = estimate_cn0_from_prompt(prompt_complex)

    if "CN0_SNV_dB_Hz" in g.columns:
        cn0_original = g["CN0_SNV_dB_Hz"].replace(0, np.nan)
        cn0_original_mean = cn0_original.mean()
        cn0_original_std = cn0_original.std()
        cn0_original_median = cn0_original.median()
        cn0_original_min = cn0_original.min()
        cn0_original_max = cn0_original.max()
    else:
        cn0_original_mean = np.nan
        cn0_original_std = np.nan
        cn0_original_median = np.nan
        cn0_original_min = np.nan
        cn0_original_max = np.nan

    return {
        "source_file": source_file,
        "tracking_file": tracking_file,
        "label": label,
        "binary_label": binary_label,
        "is_spoofed": is_spoofed,
        "tracking_valid": 1,

        "prn": int(prn),
        "n_epochs": len(g),

        "window_id": window_id,
        "window_start_row": window_start,
        "window_end_row": window_end,
        "window_size": len(g),

        # CN0
        "cn0_original_mean": cn0_original_mean,
        "cn0_original_std": cn0_original_std,
        "cn0_original_median": cn0_original_median,
        "cn0_original_min": cn0_original_min,
        "cn0_original_max": cn0_original_max,
        "cn0_est_prompt": cn0_est,

        # Prompt
        "prompt_abs_mean": P.mean(),
        "prompt_abs_std": P.std(),
        "prompt_abs_median": P.median(),
        "prompt_abs_min": P.min(),
        "prompt_abs_max": P.max(),
        "prompt_i_mean": g["Prompt_I"].mean(),
        "prompt_i_std": g["Prompt_I"].std(),
        "prompt_q_mean": g["Prompt_Q"].mean(),
        "prompt_q_std": g["Prompt_Q"].std(),

        # Doppler
        "doppler_mean": safe_mean(g, "carrier_doppler_hz"),
        "doppler_std": safe_std(g, "carrier_doppler_hz"),
        "doppler_min": safe_min(g, "carrier_doppler_hz"),
        "doppler_max": safe_max(g, "carrier_doppler_hz"),
        "doppler_range": safe_max(g, "carrier_doppler_hz") - safe_min(g, "carrier_doppler_hz")
        if "carrier_doppler_hz" in g.columns else np.nan,

        # PLL / carrier
        "pll_lock_ratio": (g["carrier_lock_test"] == 1).mean() if "carrier_lock_test" in g.columns else np.nan,
        "carrier_error_mean": safe_mean(g, "carr_error_hz"),
        "carrier_error_std": safe_std(g, "carr_error_hz"),
        "carrier_error_abs_mean": safe_abs_mean(g, "carr_error_hz"),
        "carrier_phase_std": safe_std(g, "acc_carrier_phase_rad"),

        # Kept for debugging, usually redundant with Doppler in your output.
        "carrier_error_filt_mean": safe_mean(g, "carr_error_filt_hz"),
        "carrier_error_filt_std": safe_std(g, "carr_error_filt_hz"),

        # DLL / code
        "code_error_mean": safe_mean(g, "code_error_chips"),
        "code_error_std": safe_std(g, "code_error_chips"),
        "code_error_abs_mean": safe_abs_mean(g, "code_error_chips"),
        "code_error_filt_mean": safe_mean(g, "code_error_filt_chips"),
        "code_error_filt_std": safe_std(g, "code_error_filt_chips"),
        "code_error_filt_abs_mean": safe_abs_mean(g, "code_error_filt_chips"),
        "code_freq_mean": safe_mean(g, "code_freq_chips"),
        "code_freq_std": safe_std(g, "code_freq_chips"),

        # Correlator distortion
        "early_late_balance_mean": early_late_balance.mean(),
        "early_late_balance_std": early_late_balance.std(),
        "early_late_balance_median": early_late_balance.median(),
        "early_late_balance_min": early_late_balance.min(),
        "early_late_balance_max": early_late_balance.max(),

        "early_late_asymmetry_mean": early_late_asymmetry.mean(),
        "early_late_asymmetry_std": early_late_asymmetry.std(),
        "early_late_asymmetry_median": early_late_asymmetry.median(),
        "early_late_asymmetry_p95": early_late_asymmetry.quantile(0.95),
        "early_late_asymmetry_max": early_late_asymmetry.max(),

        "prompt_dominance_mean": prompt_dominance.mean(),
        "prompt_dominance_std": prompt_dominance.std(),
        "prompt_dominance_median": prompt_dominance.median(),
        "prompt_dominance_p95": prompt_dominance.quantile(0.95),
        "prompt_dominance_max": prompt_dominance.max(),
    }


def extract_features_from_tracking_csv(path: Path, label: str, source_file: str):
    df = pd.read_csv(path)
    df = clean_tracking_df(df)

    if len(df) < MIN_TRACKING_ROWS:
        return []

    rows = []

    for prn, g in df.groupby("PRN"):
        if len(g) < MIN_TRACKING_ROWS:
            continue

        g = g.copy().reset_index(drop=True)
        window_id = 0

        for start in range(0, len(g), WINDOW_STEP):
            end = start + WINDOW_SIZE
            window = g.iloc[start:end].copy()

            if len(window) < MIN_WINDOW_ROWS:
                continue

            row = build_feature_row(
                g=window,
                label=label,
                source_file=source_file,
                tracking_file=path.name,
                prn=prn,
                window_id=window_id,
                window_start=start,
                window_end=min(end, len(g)),
            )

            rows.append(row)
            window_id += 1

    return rows


def extract_features_for_dataset(csv_dir: Path, label: str, source_file: str, features_output_path: Path):
    all_rows = []

    print("\n" + "=" * 80)
    print(f"EXTRACTING WINDOW FEATURES: {label}")
    print(f"Input CSV dir: {csv_dir}")
    print("=" * 80)

    tracking_files = sorted(csv_dir.glob("tracking*.csv"))

    if not tracking_files:
        print("No tracking*.csv files found.")
        return pd.DataFrame()

    for path in tracking_files:
        print(f"Reading: {path}")

        rows = extract_features_from_tracking_csv(
            path=path,
            label=label,
            source_file=source_file,
        )

        if rows:
            all_rows.extend(rows)
        else:
            print("  skipped, probably invalid/empty tracking")

    features_df = pd.DataFrame(all_rows)

    features_output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_output_path, index=False)

    print(f"\nSaved features: {features_output_path}")
    print(f"Shape: {features_df.shape}")

    if len(features_df) > 0:
        print(features_df.head())

    return features_df


# ============================================================
# 5. OPTIONAL OBSERVABLES FEATURES
# ============================================================

def extract_observables_features(csv_dir: Path, label: str, source_file: str):
    path = csv_dir / "observables.csv"

    if not path.exists():
        return pd.DataFrame()

    obs = pd.read_csv(path)

    for col in obs.columns:
        obs[col] = pd.to_numeric(obs[col], errors="coerce")

    if "PRN" not in obs.columns:
        return pd.DataFrame()

    if "Flag_valid_pseudorange" in obs.columns:
        obs_valid = obs[obs["Flag_valid_pseudorange"] == 1].copy()
    else:
        obs_valid = obs.copy()

    if len(obs_valid) == 0:
        return pd.DataFrame()

    rows = []
    binary_label, is_spoofed = to_binary_label(label)

    for prn, g in obs_valid.groupby("PRN"):
        row = {
            "source_file": source_file,
            "label": label,
            "binary_label": binary_label,
            "is_spoofed": is_spoofed,
            "prn": int(prn),
            "obs_n_epochs": len(g),
            "obs_doppler_mean": safe_mean(g, "Carrier_Doppler_hz"),
            "obs_doppler_std": safe_std(g, "Carrier_Doppler_hz"),
            "obs_pseudorange_mean": safe_mean(g, "Pseudorange_m"),
            "obs_pseudorange_std": safe_std(g, "Pseudorange_m"),
            "obs_carrier_phase_mean": safe_mean(g, "Carrier_phase_cycles"),
            "obs_carrier_phase_std": safe_std(g, "Carrier_phase_cycles"),
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. REPORTS
# ============================================================

def save_reports(all_features_df: pd.DataFrame):
    report_per_window_path = FEATURES_ROOT / "report_per_window.csv"
    all_features_df.to_csv(report_per_window_path, index=False)
    print(f"Saved per-window report: {report_per_window_path}")

    numeric_cols = all_features_df.select_dtypes(include=[np.number]).columns.tolist()

    exclude_from_agg = [
        "tracking_valid",
        "is_spoofed",
        "prn",
        "window_id",
        "window_start_row",
        "window_end_row",
    ]

    agg_cols = [c for c in numeric_cols if c not in exclude_from_agg]

    agg_dict = {
        "window_id": "count",
        "prn": "nunique",
    }

    for col in agg_cols:
        agg_dict[col] = ["mean", "std", "median", "min", "max"]

    report_per_file = (
        all_features_df
        .groupby(["source_file", "label", "binary_label", "is_spoofed"])
        .agg(agg_dict)
    )

    # Flatten MultiIndex columns.
    report_per_file.columns = [
        "n_windows" if col[0] == "window_id" else
        "n_prns" if col[0] == "prn" else
        f"{col[0]}_{col[1]}"
        for col in report_per_file.columns
    ]

    report_per_file = report_per_file.reset_index()

    report_per_file_path = FEATURES_ROOT / "report_per_file.csv"
    report_per_file.to_csv(report_per_file_path, index=False)

    print(f"Saved per-file report: {report_per_file_path}")
    print(report_per_file)


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)

    all_features = []

    for label, info in DATASETS.items():
        display_name = info["display_name"]
        bin_path = info["file"]

        print("\n" + "#" * 100)
        print(f"DATASET: {display_name}")
        print(f"LABEL:   {label}")
        print(f"FILE:    {bin_path}")
        print("#" * 100)

        dataset_output_dir = OUTPUT_ROOT / label
        dataset_csv_dir = dataset_output_dir / "csv_output"
        dataset_features_path = FEATURES_ROOT / f"features_{label}_windows.csv"

        # 1. Run GNSS-SDR Docker.
        run_gnss_sdr(
            label=label,
            bin_path=bin_path,
            output_dir=dataset_output_dir,
        )

        # 2. Optionally inspect .mat structure.
        if INSPECT_MAT_FILES:
            inspect_mat_files(dataset_output_dir)

        # 3. Convert .mat to .csv.
        convert_mat_to_csv(
            output_dir=dataset_output_dir,
            csv_dir=dataset_csv_dir,
        )

        # 4. Extract window tracking features.
        features_df = extract_features_for_dataset(
            csv_dir=dataset_csv_dir,
            label=label,
            source_file=bin_path.name,
            features_output_path=dataset_features_path,
        )

        if len(features_df) > 0:
            all_features.append(features_df)

        # 5. Optional observables features.
        obs_features_df = extract_observables_features(
            csv_dir=dataset_csv_dir,
            label=label,
            source_file=bin_path.name,
        )

        if len(obs_features_df) > 0:
            obs_output_path = FEATURES_ROOT / f"observables_features_{label}.csv"
            obs_features_df.to_csv(obs_output_path, index=False)
            print(f"Saved observables features: {obs_output_path}")

    # 6. Merge all tracking features.
    if all_features:
        all_features_df = pd.concat(all_features, ignore_index=True)

        all_output_path = FEATURES_ROOT / "all_gnss_window_features.csv"
        all_features_df.to_csv(all_output_path, index=False)

        print("\n" + "=" * 80)
        print("ALL WINDOW FEATURES SAVED")
        print("=" * 80)
        print(f"Saved: {all_output_path}")
        print(f"Shape: {all_features_df.shape}")

        print("\nRows per label:")
        print(all_features_df.groupby("label").size())

        print("\nRows per binary label:")
        print(all_features_df.groupby("binary_label").size())

        print("\nRows per file:")
        print(all_features_df.groupby(["source_file", "label"]).size())

        print("\nPRNs per file:")
        print(all_features_df.groupby(["source_file", "label"])["prn"].nunique())

        print("\nBasic summary by binary label:")
        summary_cols = [
            "cn0_original_mean",
            "doppler_std",
            "doppler_range",
            "carrier_error_abs_mean",
            "code_error_abs_mean",
            "early_late_asymmetry_mean",
            "early_late_asymmetry_p95",
            "prompt_dominance_mean",
        ]

        available_summary_cols = [c for c in summary_cols if c in all_features_df.columns]

        if available_summary_cols:
            print(all_features_df.groupby("binary_label")[available_summary_cols].mean())

        save_reports(all_features_df)

    else:
        print("No features extracted.")


if __name__ == "__main__":
    main()
