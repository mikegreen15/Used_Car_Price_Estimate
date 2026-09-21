#!/usr/bin/env python3
"""Presentation demo using a 600-row holdout from car_price_train.csv.

Keep this file beside used_car_price_model.py and car_price_train.csv.  The
demo hides 600 known prices, trains on the remaining 1,200 rows, writes the
same two-column prediction format required in the final competition, and then
reports accuracy because the held-out labels are available to us.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

import used_car_price_model as core


# For the current presentation, this is the only input file required.
TRAIN_PATH = Path("car_price_train.csv")
OUTPUT_PATH = Path("demo_predictions.csv")

HOLDOUT_ROWS = 600
SPLIT_SEED = 20260921


def main() -> None:
    rows, columns = core.read_csv_rows(TRAIN_PATH)
    core.require_columns(
        columns,
        [core.ID_COLUMN, core.TARGET_COLUMN]
        + core.NUMERIC_COLUMNS
        + core.CATEGORICAL_COLUMNS,
        "Training CSV",
    )
    core.validate_ids(rows, "Training CSV")

    if len(rows) <= HOLDOUT_ROWS:
        raise ValueError(
            f"Need more than {HOLDOUT_ROWS} rows, but the training CSV has {len(rows)}"
        )

    target = np.asarray(
        [core.parse_number(row[core.TARGET_COLUMN]) for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(target).all():
        raise ValueError(f"{core.TARGET_COLUMN} contains missing/non-finite values")

    # The fixed seed makes the same 600 rows the simulated test set every run.
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(len(rows))
    holdout_indices = shuffled[:HOLDOUT_ROWS]
    training_indices = shuffled[HOLDOUT_ROWS:]
    training_rows = [rows[int(index)] for index in training_indices]
    holdout_rows = [rows[int(index)] for index in holdout_indices]
    training_target = target[training_indices]
    holdout_target = target[holdout_indices]

    builder = core.FeatureBuilder()
    builder.fit(training_rows)
    training_x = builder.transform(training_rows)
    holdout_x = builder.transform(holdout_rows)

    uncensored = training_target > core.PRICE_FLOOR + 1.0e-9
    model = core.ManualLinearRegressor(l2_strength=core.DEFAULT_L2_STRENGTH)
    model.fit(training_x[uncensored], training_target[uncensored])
    predictions = np.maximum(core.PRICE_FLOOR, model.predict(holdout_x))

    # Use the exact final-submission schema, even though this is a demo split.
    core.write_predictions(OUTPUT_PATH, holdout_rows, predictions)

    errors = predictions - holdout_target
    rmse = math.sqrt(float(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    target_mean = float(np.mean(holdout_target))
    total_variation = float(np.sum((holdout_target - target_mean) ** 2))
    r_squared = 1.0 - float(np.sum(errors**2)) / total_variation

    # Re-open the deliverable to verify its physical row count and exact header.
    with OUTPUT_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
        expected_header = [core.ID_COLUMN, "sale_price_usd_pred"]
        if reader.fieldnames != expected_header:
            raise RuntimeError(f"Unexpected output header: {reader.fieldnames}")
        if len(written) != HOLDOUT_ROWS:
            raise RuntimeError("Output does not contain exactly 600 prediction rows")

    print("Used-car price model demonstration")
    print(f"Training rows: {len(training_rows)}")
    print(f"Simulated unknown-target rows: {len(holdout_rows)}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Learning rate: {core.LEARNING_RATE}")
    print(f"Maximum iterations: {core.TRAINING_STEPS}")
    print(f"Gradient stop tolerance: {core.GRADIENT_TOLERANCE:.1e}")
    print(f"L2 strength: {model.l2_strength}")
    print(f"Iterations used: {model.iterations_used}")
    print(f"Final maximum absolute gradient: {model.final_max_gradient:.3e}")
    print(f"Tolerance reached: {'yes' if model.converged else 'no; maximum iterations reached'}")
    print(f"Wrote: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
