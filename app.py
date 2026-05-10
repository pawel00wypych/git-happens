from pathlib import Path
from contextlib import asynccontextmanager
import threading

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from datetime import datetime
from opensky_worker import run_opensky_worker


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"

GENERATED_DIR = STATIC_DIR / "generated"
ARCHIVED_MAPS_DIR = STATIC_DIR / "archived_maps"

LIVE_MAP_FILE = GENERATED_DIR / "opensky_live_map.html"
PREDICTIONS_FILE = DATA_DIR / "opensky_latest_predictions.csv"

# Na czas developmentu / rate limitu ustaw False.
# Gdy chcesz, żeby aplikacja sama pobierała OpenSky po starcie, ustaw True.
ENABLE_OPENSKY_WORKER = False


# ============================================================
# DIRECTORIES
# ============================================================

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVED_MAPS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BACKGROUND WORKER STATE
# ============================================================

worker_thread = None
stop_event = threading.Event()


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_thread

    if ENABLE_OPENSKY_WORKER:
        stop_event.clear()

        worker_thread = threading.Thread(
            target=run_opensky_worker,
            kwargs={"stop_event": stop_event},
            daemon=True,
        )

        worker_thread.start()
        print("OpenSky background worker started.")
    else:
        print("OpenSky background worker disabled.")

    yield

    if ENABLE_OPENSKY_WORKER:
        stop_event.set()
        print("Stopping OpenSky background worker...")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ADS-B Anomaly Detection Web API",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ============================================================
# PAGES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="layout.html",
        context={
            "live_map_exists": LIVE_MAP_FILE.exists(),
            "worker_enabled": ENABLE_OPENSKY_WORKER,
            "worker_alive": worker_thread.is_alive() if worker_thread else False,
        },
    )


@app.get("/map", response_class=HTMLResponse)
def map_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="map.html",
        context={
            "map_exists": LIVE_MAP_FILE.exists(),
            "map_url": "/static/generated/opensky_live_map.html",
            "worker_enabled": ENABLE_OPENSKY_WORKER,
            "worker_alive": worker_thread.is_alive() if worker_thread else False,
        },
    )


@app.get("/maps", response_class=HTMLResponse)
def static_maps_page(request: Request):
    maps = []

    for path in sorted(ARCHIVED_MAPS_DIR.glob("*.html")):
        maps.append({
            "name": path.stem.replace("_", " ").replace("-", " ").title(),
            "filename": path.name,
            "url": f"/static/archived_maps/{path.name}",
        })

    return templates.TemplateResponse(
        request=request,
        name="static_maps.html",
        context={
            "maps": maps,
        },
    )


# ============================================================
# API
# ============================================================

@app.get("/api/status")
def api_status():
    return {
        "live_map_exists": LIVE_MAP_FILE.exists(),
        "live_map_path": str(LIVE_MAP_FILE),
        "predictions_exists": PREDICTIONS_FILE.exists(),
        "predictions_path": str(PREDICTIONS_FILE),
        "worker_enabled": ENABLE_OPENSKY_WORKER,
        "worker_alive": worker_thread.is_alive() if worker_thread else False,
    }


@app.get("/api/predictions/summary")
def predictions_summary():
    if not PREDICTIONS_FILE.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": "No predictions file found yet.",
                "expected_path": str(PREDICTIONS_FILE),
            },
        )

    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not read predictions file.",
                "details": str(e),
            },
        )

    total = len(df)

    if total == 0:
        return {
            "total_aircraft": 0,
            "predicted_anomalies": 0,
            "anomaly_rate": 0,
            "average_attack_probability": 0,
            "max_attack_probability": 0,
        }

    predicted_anomalies = (
        int(df["predicted_anomaly"].sum())
        if "predicted_anomaly" in df.columns
        else 0
    )

    avg_probability = (
        float(df["attack_probability"].mean())
        if "attack_probability" in df.columns
        else 0
    )

    max_probability = (
        float(df["attack_probability"].max())
        if "attack_probability" in df.columns
        else 0
    )

    return {
        "total_aircraft": total,
        "predicted_anomalies": predicted_anomalies,
        "anomaly_rate": predicted_anomalies / total,
        "average_attack_probability": avg_probability,
        "max_attack_probability": max_probability,
    }


@app.get("/api/predictions")
def api_predictions():
    if not PREDICTIONS_FILE.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": "No predictions file found yet.",
                "expected_path": str(PREDICTIONS_FILE),
            },
        )

    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not read predictions file.",
                "details": str(e),
            },
        )

    return {
        "count": len(df),
        "records": df.to_dict(orient="records"),
    }


@app.get("/api/predictions/top")
def api_top_predictions(limit: int = 20):
    if not PREDICTIONS_FILE.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": "No predictions file found yet.",
                "expected_path": str(PREDICTIONS_FILE),
            },
        )

    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not read predictions file.",
                "details": str(e),
            },
        )

    if "attack_probability" not in df.columns:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Column attack_probability not found in predictions file."
            },
        )

    top_df = df.sort_values("attack_probability", ascending=False).head(limit)

    return {
        "count": len(top_df),
        "records": top_df.to_dict(orient="records"),
    }

BIN_DASHBOARD_FILE = GENERATED_DIR / "bin_signal_comparison.html"

@app.get("/bin-dashboard", response_class=HTMLResponse)
def bin_dashboard_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="bin_dashboard.html",
        context={
            "dashboard_exists": BIN_DASHBOARD_FILE.exists(),
            "dashboard_url": "/static/generated/bin_signal_comparison.html",
        },
    )

MODEL_DASHBOARD_FILE = GENERATED_DIR / "model_dashboard.html"

@app.get("/model-dashboard", response_class=HTMLResponse)
def model_dashboard_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="model_dashboard.html",
        context={
            "dashboard_exists": MODEL_DASHBOARD_FILE.exists(),
            "dashboard_url": "/static/generated/model_dashboard.html",
            "cache_buster": int(datetime.now().timestamp()),
        },
    )