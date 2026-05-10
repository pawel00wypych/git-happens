from pathlib import Path

import pandas as pd
import numpy as np
import folium


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "av_gps"
OUTPUT_CSV = BASE_DIR / "data" / "real_gps_track.csv"

OUTPUT_MAP_DIR = BASE_DIR / "static" / "archived_maps"
OUTPUT_MAP = OUTPUT_MAP_DIR / "av_gps_spoofing_map.html"


# Jeśli chcesz czytać konkretny plik, wpisz tutaj nazwę.
# Jeśli None, skrypt wczyta wszystkie CSV z data/av_gps/.
#SINGLE_FILE = None
SINGLE_FILE = "AV-GPS-Dataset-1.csv"


# Maksymalna liczba punktów rysowanych na mapie z jednego pliku.
# Przy dużych CSV mapa może być ciężka, więc ograniczamy do demo.
NORMAL_POINTS = 10
SPOOF_POINTS = 10
MARKER_EVERY_N = 10


def normalize_col_name(col: str) -> str:
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_map = {
        normalize_col_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_col_name(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    return None


def read_av_gps_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    print(f"\nReading: {path.name}")
    print("Original columns:")
    print(list(df.columns))

    lat_col = find_column(df, [
        "latitude",
        "lat",
        "gps_latitude",
        "gps_lat",
    ])

    lon_col = find_column(df, [
        "longitude",
        "lon",
        "lng",
        "gps_longitude",
        "gps_lon",
    ])

    altitude_col = find_column(df, [
        "altitude",
        "alt",
        "gps_altitude",
    ])

    speed_col = find_column(df, [
        "velocity",
        "speed",
        "gps_speed",
    ])

    heading_col = find_column(df, [
        "heading",
        "course",
        "yaw",
        "gps_heading",
    ])

    label_col = find_column(df, [
        "data_type",
        "datatype",
        "label",
        "is_spoofed",
        "spoofed",
        "attack",
    ])

    time_col = find_column(df, [
        "timestamp",
        "time",
        "gps_time",
        "datetime",
        "date_time",
    ])

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not find latitude/longitude columns in {path.name}. "
            f"Columns: {list(df.columns)}"
        )

    out = pd.DataFrame()

    # Timestamp
    if time_col is not None:
        out["timestamp"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

        # Jeśli timestamp nie dał się sparsować, robimy indeks czasowy.
        if out["timestamp"].isna().all():
            out["timestamp"] = pd.date_range(
                start="2026-01-01T00:00:00Z",
                periods=len(df),
                freq="1s",
            )
    else:
        out["timestamp"] = pd.date_range(
            start="2026-01-01T00:00:00Z",
            periods=len(df),
            freq="1s",
        )

    out["device_id"] = path.stem

    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce")

    if altitude_col is not None:
        out["altitude"] = pd.to_numeric(df[altitude_col], errors="coerce")
    else:
        out["altitude"] = np.nan

    if speed_col is not None:
        out["speed"] = pd.to_numeric(df[speed_col], errors="coerce")
    else:
        out["speed"] = np.nan

    if heading_col is not None:
        out["heading"] = pd.to_numeric(df[heading_col], errors="coerce")
    else:
        out["heading"] = np.nan

    # Label: w AV-GPS-Dataset Data Type: 0 = normal, 1 = spoofing
    if label_col is not None:
        out["is_spoofed"] = (
            pd.to_numeric(df[label_col], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        # Jeżeli plik ma "Normal-Data" w nazwie, traktujemy jako clean.
        # Pozostałe pliki traktujemy jako spoofing/attack.
        if "normal" in path.name.lower():
            out["is_spoofed"] = 0
        else:
            out["is_spoofed"] = 1

    out["source_file"] = path.name

    out = out.dropna(subset=["timestamp", "lat", "lon"])
    out = out.sort_values("timestamp")

    print("Normalized columns:")
    print(list(out.columns))
    print("Rows:", len(out))
    print("Spoofing distribution:")
    print(out["is_spoofed"].value_counts(dropna=False))

    return out


def downsample_for_map(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """
    Ogranicza liczbę punktów na mapie, żeby HTML nie był ogromny.
    Zachowuje równomiernie rozłożone punkty.
    """
    if len(df) <= max_points:
        return df.copy()

    indices = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[indices].copy()


def create_spoofing_map(df: pd.DataFrame) -> folium.Map:
    map_df = df.dropna(subset=["lat", "lon"]).copy()

    if map_df.empty:
        raise ValueError("No valid lat/lon rows for map.")

    # Bierzemy jeden scenariusz/plik, żeby mapa była czytelna.
    # Preferujemy plik, który zawiera spoofing.
    spoof_devices = (
        map_df.groupby("device_id")["is_spoofed"]
        .max()
        .sort_values(ascending=False)
    )

    selected_device = spoof_devices.index[0]
    track = map_df[map_df["device_id"] == selected_device].copy()
    track = track.sort_values("timestamp")

    normal_df = track[track["is_spoofed"] == 0].copy()
    spoof_df = track[track["is_spoofed"] == 1].copy()

    # Jeżeli plik zawiera tylko spoofing, bierzemy początkowy fragment jako tło/normal-like.
    if normal_df.empty and not spoof_df.empty:
        split_idx = max(1, int(len(track) * 0.35))
        normal_df = track.iloc[:split_idx].copy()
        spoof_df = track.iloc[split_idx:].copy()
        normal_df["is_spoofed"] = 0
        spoof_df["is_spoofed"] = 1

    normal_df = downsample_for_map(normal_df, NORMAL_POINTS)
    spoof_df = downsample_for_map(spoof_df, SPOOF_POINTS)

    combined_for_center = pd.concat([normal_df, spoof_df], ignore_index=True)

    center_lat = combined_for_center["lat"].mean()
    center_lon = combined_for_center["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles="OpenStreetMap",
    )

    normal_points = normal_df[["lat", "lon"]].values.tolist()
    spoof_points = spoof_df[["lat", "lon"]].values.tolist()

    # Normal trajectory
    if len(normal_points) >= 2:
        folium.PolyLine(
            locations=normal_points,
            color="green",
            weight=6,
            opacity=0.85,
            tooltip="Normal GPS trajectory",
        ).add_to(m)

    # Spoofed trajectory
    if len(spoof_points) >= 2:
        folium.PolyLine(
            locations=spoof_points,
            color="red",
            weight=6,
            opacity=0.9,
            dash_array="10, 8",
            tooltip="Spoofed / anomalous GPS trajectory",
        ).add_to(m)

    # Connection between last normal and first spoofed point
    if len(normal_points) > 0 and len(spoof_points) > 0:
        folium.PolyLine(
            locations=[normal_points[-1], spoof_points[0]],
            color="orange",
            weight=4,
            opacity=0.9,
            dash_array="5, 8",
            tooltip="Transition to spoofing",
        ).add_to(m)

    # Numbered/selected markers only every N points
    def add_sparse_markers(points_df: pd.DataFrame, color: str, status: str):
        for i, (_, row) in enumerate(points_df.iterrows()):
            if i % MARKER_EVERY_N != 0 and i != len(points_df) - 1:
                continue

            popup = f"""
            <b>Status:</b> {status}<br>
            <b>Device/file:</b> {row.get("device_id", "")}<br>
            <b>Source file:</b> {row.get("source_file", "")}<br>
            <hr>
            <b>Timestamp:</b> {row.get("timestamp", "")}<br>
            <b>Latitude:</b> {row.get("lat", "")}<br>
            <b>Longitude:</b> {row.get("lon", "")}<br>
            <b>Altitude:</b> {row.get("altitude", "")}<br>
            <b>Speed:</b> {row.get("speed", "")}<br>
            <b>Heading:</b> {row.get("heading", "")}<br>
            <b>is_spoofed:</b> {row.get("is_spoofed", "")}
            """

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=popup,
                tooltip=f"{status} | point {i}",
            ).add_to(m)

    add_sparse_markers(normal_df, "green", "Normal GPS data")
    add_sparse_markers(spoof_df, "red", "GPS spoofing attack")

    # Start marker
    if len(normal_df) > 0:
        first = normal_df.iloc[0]
        folium.Marker(
            location=[first["lat"], first["lon"]],
            popup="Start of normal GPS trajectory",
            tooltip="Start",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)

    # Spoofing start marker
    if len(spoof_df) > 0:
        first_spoof = spoof_df.iloc[0]

        folium.Marker(
            location=[first_spoof["lat"], first_spoof["lon"]],
            popup="Spoofing starts here",
            tooltip="Spoofing starts here",
            icon=folium.Icon(color="orange", icon="warning-sign"),
        ).add_to(m)

        # Highlight anomaly start
        folium.Circle(
            location=[first_spoof["lat"], first_spoof["lon"]],
            radius=25,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.15,
            popup="Suspicious GPS trajectory deviation starts here",
        ).add_to(m)

    # End marker
    if len(spoof_df) > 0:
        last = spoof_df.iloc[-1]
        folium.Marker(
            location=[last["lat"], last["lon"]],
            popup="Final spoofed GPS position",
            tooltip="Final spoofed GPS position",
            icon=folium.Icon(color="red", icon="remove"),
        ).add_to(m)

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 390px;
        background-color: white;
        border: 2px solid grey;
        z-index: 9999;
        font-size: 14px;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    ">
        <b>GPS Spoofing Trajectory Demo</b><br>
        <span style="color:green;">━</span> Normal GPS trajectory<br>
        <span style="color:red;">━ ━</span> Spoofed / anomalous GPS trajectory<br>
        <span style="color:orange;">- - -</span> Transition to spoofing<br>
        <hr>
        Dataset: AV-GPS<br>
        Selected device/file: {selected_device}<br>
        Normal points shown: {len(normal_df)}<br>
        Spoofed points shown: {len(spoof_df)}
    </div>
    """

    callout_html = """
    <div style="
        position: fixed;
        top: 90px;
        right: 40px;
        width: 340px;
        background-color: white;
        border: 2px solid #cc0000;
        z-index: 9999;
        font-size: 15px;
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    ">
        <b style="color:#cc0000; font-size:18px;">⚠ Spoofing anomaly</b><br>
        Reported GPS trajectory deviates from the expected path.
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(callout_html))

    return m


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Missing input directory: {INPUT_DIR}\n"
            f"Create it and put AV-GPS CSV files there."
        )

    if SINGLE_FILE:
        files = [INPUT_DIR / SINGLE_FILE]
    else:
        files = sorted(INPUT_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {INPUT_DIR}"
        )

    all_tracks = []

    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        track_df = read_av_gps_csv(path)
        all_tracks.append(track_df)

    combined = pd.concat(all_tracks, ignore_index=True)

    combined = combined.dropna(subset=["lat", "lon"])
    combined = combined.sort_values(["device_id", "timestamp"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    OUTPUT_MAP_DIR.mkdir(parents=True, exist_ok=True)

    m = create_spoofing_map(combined)
    m.save(str(OUTPUT_MAP))

    print("\n====================================")
    print("Saved CSV:", OUTPUT_CSV)
    print("Saved map:", OUTPUT_MAP)
    print("Total rows:", len(combined))
    print("Devices/files:", combined["device_id"].nunique())
    print("Final spoofing distribution:")
    print(combined["is_spoofed"].value_counts(dropna=False))
    print("Preview:")
    print(combined.head())


if __name__ == "__main__":
    main()