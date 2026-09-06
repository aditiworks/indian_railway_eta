"""Local FastAPI service for railway ETA prediction and platform conflicts.

Run locally from this directory after placing model.pkl beside this file:
    python railway_eta_api.py

Install dependencies if needed:
    python -m pip install fastapi "uvicorn[standard]" pandas lightgbm scikit-learn
"""

from __future__ import annotations

import os
import pickle
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# This default resolves beside this script on Windows, Linux, and Render/Railway.
# MODEL_PATH may optionally override it for a mounted model volume.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(SCRIPT_DIRECTORY / "model.pkl")))
MAX_RECENT_ARRIVALS = 1_000
CONFLICT_WINDOW_MINUTES = 10

# Render/Railway can provide the deployed frontend URL. During initial setup,
# the '*' fallback permits requests from any origin; set FRONTEND_URL in
# production to the exact deployed frontend domain for a narrower policy.
FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")
CORS_ORIGINS = [FRONTEND_URL]
LOCALHOST_ORIGIN_REGEX = r"^http://localhost(?::\d+)?$"

# These values match the synthetic eight-station route. In production, persist
# these station averages alongside the model instead of keeping them in code.
STATION_ALIASES = {
    "NDLS": "New Delhi",
    "CNB": "Kanpur Central",
    "PRYJ": "Prayagraj Junction",
    "DDU": "Pt. Deen Dayal Upadhyaya Junction",
    "GAYA": "Gaya Junction",
    "DHN": "Dhanbad Junction",
    "ASN": "Asansol Junction",
    "HWH": "Howrah Junction",
}
HISTORICAL_STATION_DELAY_MINUTES = {
    "New Delhi": 11.03,
    "Kanpur Central": 11.47,
    "Prayagraj Junction": 10.98,
    "Pt. Deen Dayal Upadhyaya Junction": 12.14,
    "Gaya Junction": 12.22,
    "Dhanbad Junction": 13.03,
    "Asansol Junction": 13.78,
    "Howrah Junction": 14.32,
}
DEFAULT_HISTORICAL_DELAY_MINUTES = 12.37

model: object | None = None
model_load_error: str | None = None
recent_arrivals: list[dict[str, object]] = []
recent_arrivals_lock = threading.Lock()


class PredictETARequest(BaseModel):
    train_id: str
    station: str = Field(description="Station code (e.g. NDLS) or station name.")
    current_delay_min: float
    weather_condition: str
    distance_remaining_km: float = Field(ge=0)
    platform_number: str | int | None = Field(
        default=None,
        description="Optional assigned platform. Required only for automatic conflict checking.",
    )
    # Optional: without this, the returned timestamp is now + predicted delay,
    # not timetable arrival + predicted delay.
    scheduled_arrival_time: datetime | None = None


class ArrivalPrediction(BaseModel):
    train_id: str
    station: str
    platform: str
    predicted_arrival_time: datetime


def as_utc(value: datetime) -> datetime:
    """Normalize user-supplied timestamps for safe time-difference comparisons."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def model_feature_frame(request: PredictETARequest) -> pd.DataFrame:
    """Create exactly the engineered columns used by train_eta_lightgbm.py."""
    station_name = STATION_ALIASES.get(request.station.upper(), request.station)
    now = datetime.now(timezone.utc)
    candidate_features = {
        "train_id": request.train_id,
        "station": station_name,
        "weather_condition": request.weather_condition,
        "scheduled_arrival_hour": now.hour,
        "scheduled_arrival_day_of_week": now.weekday(),
        "distance_remaining_km": request.distance_remaining_km,
        "current_cumulative_delay_min": request.current_delay_min,
        "historical_avg_delay_for_this_station": HISTORICAL_STATION_DELAY_MINUTES.get(
            station_name, DEFAULT_HISTORICAL_DELAY_MINUTES
        ),
    }

    # A scikit-learn Pipeline exposes feature_names_in_ after fitting. Selecting
    # by that schema catches a mismatch between this API and a different model.
    required_features = list(getattr(model, "feature_names_in_", candidate_features.keys()))
    missing = set(required_features).difference(candidate_features)
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Model expects unsupported feature(s): {sorted(missing)}. Re-export a compatible model.",
        )
    return pd.DataFrame([{column: candidate_features[column] for column in required_features}])


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the saved pipeline once; keep the app alive to report useful errors."""
    global model, model_load_error
    try:
        with MODEL_PATH.open("rb") as model_file:
            model = pickle.load(model_file)
        model_load_error = None
    except (FileNotFoundError, pickle.UnpicklingError, ImportError, ModuleNotFoundError) as error:
        model = None
        model_load_error = str(error)
    yield


app = FastAPI(title="Indian Railways ETA API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # allow_origin_regex covers http://localhost and any localhost port.
    # Credentials must be disabled when the fallback wildcard origin is used.
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=LOCALHOST_ORIGIN_REGEX,
    allow_credentials=FRONTEND_URL != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict-eta")
def predict_eta(request: PredictETARequest) -> dict[str, object]:
    """Predict the arrival delay and return an ETA timestamp."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable at {MODEL_PATH}. {model_load_error or ''}".strip(),
        )

    predicted_delay_minutes = float(model.predict(model_feature_frame(request))[0])
    if request.scheduled_arrival_time is not None:
        predicted_eta = as_utc(request.scheduled_arrival_time) + timedelta(minutes=predicted_delay_minutes)
        eta_basis = "scheduled_arrival_time + predicted_delay_minutes"
    else:
        # The trained project model predicts arrival *delay*, while the compact
        # request has no timetable timestamp. This is a clearly-labelled local
        # estimate, not a replacement for a timetable-based ETA.
        predicted_eta = datetime.now(timezone.utc) + timedelta(minutes=predicted_delay_minutes)
        eta_basis = "current_time + predicted_delay_minutes (no scheduled_arrival_time supplied)"

    # Both endpoints share this exact global `recent_arrivals` list. Only save
    # predictions with a known platform; otherwise a platform conflict cannot
    # be assessed. Use the same lock as /conflicts for thread-safe access.
    stored_for_conflict_check = request.platform_number is not None
    if stored_for_conflict_check:
        with recent_arrivals_lock:
            platform = str(request.platform_number)
            # Upsert by train/station/platform so retries and repeated UI clicks
            # update the latest ETA instead of creating duplicate candidates.
            recent_arrivals[:] = [
                entry
                for entry in recent_arrivals
                if not (
                    entry["train_id"] == request.train_id
                    and entry["station"] == request.station
                    and entry["platform"] == platform
                )
            ]
            recent_arrivals.append(
                {
                    "train_id": request.train_id,
                    "station": request.station,
                    "platform": platform,
                    "predicted_arrival_time": predicted_eta,
                    "source": "predict_eta",
                }
            )
            del recent_arrivals[:-MAX_RECENT_ARRIVALS]

    return {
        "train_id": request.train_id,
        "station": request.station,
        "predicted_delay_minutes": round(predicted_delay_minutes, 2),
        "predicted_eta": predicted_eta.isoformat(),
        "eta_basis": eta_basis,
        "stored_for_conflict_check": stored_for_conflict_check,
    }


@app.get("/conflicts")
def conflicts(
    arrivals: list[ArrivalPrediction] = Body(
        default=[], description="Currently predicted platform arrivals to check and retain in memory."
    ),
) -> dict[str, object]:
    """Find same-platform arrivals within ten minutes, including recent requests."""
    incoming = [
        {
            "train_id": arrival.train_id,
            "station": arrival.station,
            "platform": arrival.platform,
            "predicted_arrival_time": as_utc(arrival.predicted_arrival_time),
            "source": "current_request",
        }
        for arrival in arrivals
    ]

    with recent_arrivals_lock:
        # `recent_arrivals` includes platform-tagged POST /predict-eta results
        # plus any manual entries supplied to this endpoint.
        candidates = [*recent_arrivals, *incoming]
        conflicts_found: list[dict[str, object]] = []
        seen_conflict_pairs: set[tuple[str, str, str, str]] = set()
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                if (
                    left["train_id"] == right["train_id"]
                    or left["station"] != right["station"]
                    or left["platform"] != right["platform"]
                ):
                    continue
                time_difference = abs(
                    (left["predicted_arrival_time"] - right["predicted_arrival_time"]).total_seconds()
                ) / 60
                if time_difference <= CONFLICT_WINDOW_MINUTES:
                    # A canonical key makes A/B and B/A (and any stale duplicate
                    # candidates) resolve to one conflict response entry.
                    train_a, train_b = sorted((str(left["train_id"]), str(right["train_id"])))
                    conflict_key = (str(left["station"]), str(left["platform"]), train_a, train_b)
                    if conflict_key in seen_conflict_pairs:
                        continue
                    seen_conflict_pairs.add(conflict_key)
                    conflicts_found.append(
                        {
                            "platform": left["platform"],
                            "station_a": left["station"],
                            "station_b": right["station"],
                            "train_a": left["train_id"],
                            "train_b": right["train_id"],
                            "arrival_a": left["predicted_arrival_time"].isoformat(),
                            "arrival_b": right["predicted_arrival_time"].isoformat(),
                            "time_difference_minutes": round(time_difference, 2),
                            "conflict": True,
                        }
                    )

        recent_arrivals.extend(incoming)
        del recent_arrivals[:-MAX_RECENT_ARRIVALS]

    return {
        "conflict_window_minutes": CONFLICT_WINDOW_MINUTES,
        "conflicts": conflicts_found,
        "recent_predictions_stored": len(recent_arrivals),
    }


@app.delete("/clear-predictions")
def clear_predictions() -> dict[str, object]:
    """Clear the in-memory platform-arrival history."""
    with recent_arrivals_lock:
        recent_arrivals.clear()
    return {
        "message": "Prediction history cleared",
        "recent_predictions_stored": 0,
    }


if __name__ == "__main__":
    # Render/Railway supplies PORT. The local default keeps direct execution easy.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("railway_eta_api:app", host="0.0.0.0", port=port)
