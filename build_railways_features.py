"""Add leakage-safe ETA features to the synthetic Indian Railways CSV.

Example:
    python build_railways_features.py synthetic_indian_railways_running_data.csv \
        --output synthetic_indian_railways_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Fixed illustrative route distances. Replace these with route-kilometre values
# from an authoritative railway source when using real operational data.
DISTANCE_FROM_START_KM = {
    "NDLS": 0,
    "CNB": 440,
    "PRYJ": 635,
    "DDU": 785,
    "GAYA": 1_000,
    "DHN": 1_190,
    "ASN": 1_260,
    "HWH": 1_460,
}
TOTAL_ROUTE_KM = max(DISTANCE_FROM_START_KM.values())

REQUIRED_COLUMNS = {"journey_id", "station_sequence", "station_code", "arrival_delay_minutes"}


def build_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Return and save the source data with leakage-safe ETA features added."""
    data = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

    data["arrival_delay_minutes"] = pd.to_numeric(data["arrival_delay_minutes"], errors="coerce")
    if data["arrival_delay_minutes"].isna().any():
        raise ValueError("arrival_delay_minutes must be numeric and non-empty.")

    unknown_stations = sorted(set(data["station_code"]) - set(DISTANCE_FROM_START_KM))
    if unknown_stations:
        raise ValueError(
            "No fixed distance is configured for station code(s): " + ", ".join(unknown_stations)
        )

    # Exclude *all* records for the current journey at this station from its
    # historical average. This remains safe even if a journey visits a station
    # more than once, unlike simply subtracting the current row's delay.
    station_delay_sum = data.groupby("station_code")["arrival_delay_minutes"].transform("sum")
    station_delay_count = data.groupby("station_code")["arrival_delay_minutes"].transform("count")
    current_journey_station_sum = data.groupby(
        ["station_code", "journey_id"]
    )["arrival_delay_minutes"].transform("sum")
    current_journey_station_count = data.groupby(
        ["station_code", "journey_id"]
    )["arrival_delay_minutes"].transform("count")
    other_journey_count = station_delay_count - current_journey_station_count
    data["historical_avg_delay_for_this_station"] = np.where(
        other_journey_count > 0,
        (station_delay_sum - current_journey_station_sum) / other_journey_count,
        np.nan,
    )

    # Sort explicitly before shifting: a train's current known delay is the
    # previous station's observed arrival delay, and the origin has none.
    route_order = data.sort_values(["journey_id", "station_sequence"])
    previous_station_delay = route_order.groupby("journey_id")["arrival_delay_minutes"].shift()
    data["current_cumulative_delay_min"] = previous_station_delay.reindex(data.index).fillna(0.0)

    data["distance_from_start_km"] = data["station_code"].map(DISTANCE_FROM_START_KM)
    data["distance_remaining_km"] = TOTAL_ROUTE_KM - data["distance_from_start_km"]

    # Round computed values only for presentation/CSV readability.
    data["historical_avg_delay_for_this_station"] = data[
        "historical_avg_delay_for_this_station"
    ].round(3)
    data["current_cumulative_delay_min"] = data["current_cumulative_delay_min"].round(3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build synthetic railway ETA features.")
    parser.add_argument("input_csv", type=Path, help="Source synthetic running-data CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_indian_railways_features.csv"),
        help="Output CSV path (default: synthetic_indian_railways_features.csv).",
    )
    args = parser.parse_args()
    feature_data = build_features(args.input_csv, args.output)
    print(f"Created {len(feature_data):,} rows with four derived columns: {args.output.resolve()}")
    print("Added: historical_avg_delay_for_this_station, current_cumulative_delay_min,")
    print("       distance_from_start_km, distance_remaining_km")
