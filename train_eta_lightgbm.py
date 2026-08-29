"""Train and evaluate a LightGBM arrival-time (ETA) predictor.

Expected input columns:
    train_id, station, scheduled_arrival, actual_arrival,
    distance_remaining_km, current_cumulative_delay_min, weather_condition,
    historical_avg_delay_for_this_station, arrival_delay_minutes

Example:
    python train_eta_lightgbm.py path/to/running_data.csv

Install dependencies once if necessary:
    pip install pandas scikit-learn lightgbm
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


# ---- Experiment settings: keep these fixed for a reproducible pitch number. ----
TEST_SIZE = 0.20       # Fraction of records held out exclusively for evaluation.
RANDOM_STATE = 42      # Fixed split/model seed; change only for a new experiment.
N_ESTIMATORS = 500     # More trees can improve fit, but increase train time.
LEARNING_RATE = 0.04   # Smaller values are steadier; tune together with N_ESTIMATORS.
NUM_LEAVES = 31        # Main LightGBM tree-complexity control.

REQUIRED_COLUMNS = {
    "train_id",
    "station",
    "scheduled_arrival",
    "actual_arrival",
    "distance_remaining_km",
    "current_cumulative_delay_min",
    "weather_condition",
    "historical_avg_delay_for_this_station",
}

# These are the column names in synthetic_indian_railways_features.csv.  The
# trainer automatically creates the canonical names it expects from them.
DEFAULT_COLUMN_ALIASES = {
    "station": "station_name",
    "scheduled_arrival": "scheduled_arrival_time",
    "actual_arrival": "actual_arrival_time",
}


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both newer and older scikit-learn versions."""
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_and_prepare(csv_path: Path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Validate the source data and create model-ready timetable features."""
    data = pd.read_csv(csv_path)
    column_map = {**DEFAULT_COLUMN_ALIASES, **(column_map or {})}
    for canonical_name, source_name in column_map.items():
        if canonical_name not in data.columns and source_name in data.columns:
            data[canonical_name] = data[source_name]
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        available = ", ".join(data.columns)
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Input is missing required columns: {missing}.\n"
            f"Available columns: {available}"
        )

    for column in ("scheduled_arrival", "actual_arrival"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"Column '{column}' contains unparseable arrival times.")

    numeric_columns = [
        "distance_remaining_km",
        "current_cumulative_delay_min",
        "historical_avg_delay_for_this_station",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"Column '{column}' must contain numeric, non-empty values.")

    if data[["train_id", "station", "weather_condition"]].isna().any().any():
        raise ValueError("train_id, station, and weather_condition cannot be empty.")

    # Do not pass raw datetime/epoch values to the model.  These timetable
    # components preserve useful operating-time patterns while staying bounded.
    data["scheduled_arrival_hour"] = data["scheduled_arrival"].dt.hour
    data["scheduled_arrival_day_of_week"] = data["scheduled_arrival"].dt.dayofweek

    # Predict delay (minutes), not an absolute datetime.  This makes the model
    # output directly comparable with the naive current-cumulative-delay ETA.
    if "arrival_delay_minutes" not in data.columns:
        data["arrival_delay_minutes"] = (
            data["actual_arrival"] - data["scheduled_arrival"]
        ).dt.total_seconds() / 60
    data["arrival_delay_minutes"] = pd.to_numeric(data["arrival_delay_minutes"], errors="coerce")
    if data["arrival_delay_minutes"].isna().any():
        raise ValueError("arrival_delay_minutes must be numeric and non-empty.")
    return data


def print_pretraining_audit(
    data: pd.DataFrame, feature_columns: list[str], target_column: str
) -> None:
    """Print the target and every feature summary before any fitting occurs."""
    print("\n" + "=" * 72)
    print("PRE-TRAINING DATA AUDIT — MODEL HAS NOT BEEN TRAINED")
    print("=" * 72)
    print(f"Target column: {target_column}")
    print("First 5 target values:")
    print(data[target_column].head(5).to_string(index=False))
    print("\nFeature columns used by the model:")
    for feature in feature_columns:
        print(f"  - {feature}")

    print("\n.describe() for each feature:")
    for feature in feature_columns:
        print(f"\n[{feature}]")
        print(data[[feature]].describe(include="all").to_string())

    raw_datetime_columns = {"scheduled_arrival", "actual_arrival"}
    raw_datetime_features = raw_datetime_columns.intersection(feature_columns)
    epoch_features = [feature for feature in feature_columns if "epoch" in feature.lower()]
    print("\nDatetime feature check:")
    print(f"  Raw datetime columns passed to model: {sorted(raw_datetime_features) or 'None'}")
    print(f"  Epoch/timestamp-like columns passed to model: {epoch_features or 'None'}")
    print("  scheduled_arrival is used only as hour and day-of-week components.")
    print("  actual_arrival is used only to derive/validate the target; it is not a feature.")
    print("=" * 72)


def evaluate_and_train(
    csv_path: Path,
    model_path: Path,
    column_map: dict[str, str] | None = None,
    confirm_train: bool = False,
) -> None:
    data = load_and_prepare(csv_path, column_map)

    categorical_features = ["train_id", "station", "weather_condition"]
    numeric_features = [
        "scheduled_arrival_hour",
        "scheduled_arrival_day_of_week",
        "distance_remaining_km",
        "current_cumulative_delay_min",
        "historical_avg_delay_for_this_station",
    ]
    feature_columns = categorical_features + numeric_features
    target_column = "arrival_delay_minutes"
    target = data[target_column]

    # This audit always runs first. By default the program exits here, so the
    # user can inspect the exact target/features before trusting any metrics.
    print_pretraining_audit(data, feature_columns, target_column)
    if not confirm_train:
        print("Training intentionally stopped. Review this audit, then rerun with --confirm-train.")
        return

    # Import model packages only after the audit has been reviewed. This also
    # lets users inspect a CSV on a machine that has not installed LightGBM yet.
    try:
        from lightgbm import LGBMRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install them with: "
            "python -m pip install lightgbm scikit-learn pandas\n"
            f"Original error: {error}"
        ) from error

    # The split happens before fitting the encoder and model, preventing test-row
    # information from influencing their fitted state.
    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_columns], target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # This is the last checkpoint before preprocessing and fitting. It reports
    # the *actual* target object that is supplied to pipeline.fit below.
    print("\n" + "=" * 72)
    print("POST-SPLIT PRE-FIT CHECKPOINT — TRAINING HAS NOT STARTED")
    print("=" * 72)
    print(f"Target variable passed to pipeline.fit: y_train (source column: {y_train.name})")
    print(f"X_train shape: {X_train.shape}; y_train shape: {y_train.shape}")
    print(f"X_train/y_train indices identical: {X_train.index.equals(y_train.index)}")
    print("Index handling: train_test_split preserves original row indices; no reset/reindex occurs.")
    print("\ny_train.describe():")
    print(y_train.describe().to_string())
    print("\nX_train columns and dtypes before encoding:")
    print(X_train.dtypes.to_string())
    print(
        "\nString handling: train_id, station, and weather_condition are in X_train only; "
        "ColumnTransformer one-hot encodes them. They cannot become y_train."
    )
    if not X_train.index.equals(y_train.index):
        raise RuntimeError("Safety stop: X_train and y_train indices are not aligned.")
    if y_train.abs().mean() > 1_000:
        raise RuntimeError(
            "Safety stop: y_train has an implausibly large mean for minutes. "
            "Check the target source column before fitting."
        )
    print("Pre-fit target check passed; fitting pipeline with pipeline.fit(X_train, y_train).")
    print("=" * 72)

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )
    model = LGBMRegressor(
        objective="regression",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessing", preprocessing), ("lightgbm", model)])
    pipeline.fit(X_train, y_train)

    model_predictions = pipeline.predict(X_test)

    # Naive current Railway-style ETA: timetable arrival + known current delay.
    # With a delay-minutes target, its predicted delay is simply the current
    # cumulative delay. Errors equal the corresponding arrival-time errors.
    baseline_predictions = X_test["current_cumulative_delay_min"]

    comparison = pd.DataFrame(
        {
            "Method": ["Naive ETA baseline", "LightGBM model"],
            "MAE (minutes)": [
                mean_absolute_error(y_test, baseline_predictions),
                mean_absolute_error(y_test, model_predictions),
            ],
            "RMSE (minutes)": [
                mean_squared_error(y_test, baseline_predictions) ** 0.5,
                mean_squared_error(y_test, model_predictions) ** 0.5,
            ],
        }
    ).round(3)

    baseline_mae, model_mae = comparison["MAE (minutes)"]
    baseline_rmse, model_rmse = comparison["RMSE (minutes)"]
    print("\n" + "=" * 62)
    print("ETA PREDICTION EVALUATION — HELD-OUT TEST SET")
    print("=" * 62)
    print(f"Input rows: {len(data):,} | Test rows: {len(X_test):,} | Random seed: {RANDOM_STATE}")
    print(comparison.to_string(index=False))
    print("-" * 62)
    print(f"MAE improvement:  {baseline_mae - model_mae:.3f} minutes "
          f"({(1 - model_mae / baseline_mae) * 100:.1f}%)")
    print(f"RMSE improvement: {baseline_rmse - model_rmse:.3f} minutes "
          f"({(1 - model_rmse / baseline_rmse) * 100:.1f}%)")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(pipeline, model_file)
    print(f"\nSaved trained pipeline to: {model_path.resolve()}")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare ETA predictors.")
    parser.add_argument("csv_path", type=Path, help="CSV file containing the required columns.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model.pkl"),
        help="Output .pkl path for the fitted preprocessing-and-model pipeline.",
    )
    parser.add_argument(
        "--confirm-train",
        action="store_true",
        help="Train only after first reviewing the default pre-training audit.",
    )
    parser.add_argument(
        "--column-map",
        default="{}",
        help=(
            "Optional JSON mapping of required names to source names, e.g. "
            "'{\"station\": \"station_name\"}'. Synthetic aliases are automatic."
        ),
    )
    arguments = parser.parse_args()
    try:
        parsed_column_map = json.loads(arguments.column_map)
    except json.JSONDecodeError as error:
        parser.error(f"--column-map must be valid JSON: {error.msg}")
    if not isinstance(parsed_column_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in parsed_column_map.items()
    ):
        parser.error("--column-map must be a JSON object with string keys and values.")
    evaluate_and_train(
        arguments.csv_path,
        arguments.model_path,
        parsed_column_map,
        confirm_train=arguments.confirm_train,
    )
