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

import matplotlib
import numpy as np

import used_car_price_model as core

# Use a noninteractive backend so charts work from Terminal and on the
# instructor's computer without opening GUI windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# For the current presentation, this is the only input file required.
TRAIN_PATH = Path("../raw_csv/car_price_train.csv")
OUTPUT_PATH = Path("../predictions/demo_predictions.csv")
CHART_DIRECTORY = Path("../images/demo_presentation_charts")

HOLDOUT_ROWS = 600
SPLIT_SEED = 20260921


NAVY = "#17365D"
BLUE = "#4472C4"
ORANGE = "#ED7D31"
LIGHT_BLUE = "#DDEBF7"


def dollar_axis(value: float, _position: float) -> str:
    """Format chart axes as compact US-dollar values."""
    return f"${value / 1000:,.0f}k"


def save_presentation_charts(
    actual: np.ndarray,
    predicted: np.ndarray,
    rmse: float,
    mae: float,
    r_squared: float,
) -> list[Path]:
    """Save individual figures plus a one-page presentation dashboard."""
    CHART_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    formatter = FuncFormatter(dollar_axis)
    residuals = predicted - actual
    absolute_errors = np.abs(residuals)
    chart_paths: list[Path] = []

    # Chart 1: points near the diagonal indicate accurate predictions.
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    axis.scatter(actual, predicted, s=30, alpha=0.60, color=BLUE, edgecolors="none")
    axis.plot([low, high], [low, high], color=ORANGE, linewidth=2.2,
              label="Perfect prediction")
    axis.set(title="Actual vs. Predicted Sale Price",
             xlabel="Actual sale price", ylabel="Predicted sale price")
    axis.xaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_formatter(formatter)
    axis.legend(frameon=True)
    axis.text(0.03, 0.95, f"R² = {r_squared:.4f}\nRMSE = ${rmse:,.0f}",
              transform=axis.transAxes, va="top", fontsize=11,
              bbox={"facecolor": "white", "edgecolor": LIGHT_BLUE, "alpha": 0.9})
    figure.tight_layout()
    path = CHART_DIRECTORY / "01_actual_vs_predicted.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    chart_paths.append(path)

    # Chart 2: random scatter around zero is preferable to a visible pattern.
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    axis.scatter(predicted, residuals, s=30, alpha=0.60, color=BLUE,
                 edgecolors="none")
    axis.axhline(0.0, color=ORANGE, linewidth=2.0)
    axis.set(title="Residuals vs. Predicted Price",
             xlabel="Predicted sale price",
             ylabel="Residual: predicted minus actual")
    axis.xaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_formatter(formatter)
    figure.tight_layout()
    path = CHART_DIRECTORY / "02_residuals_vs_predicted.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    chart_paths.append(path)

    # Chart 3: shows the full error distribution rather than only its average.
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    axis.hist(absolute_errors, bins=25, color=BLUE, alpha=0.85,
              edgecolor="white")
    axis.axvline(mae, color=ORANGE, linewidth=2.2,
                 label=f"MAE = ${mae:,.0f}")
    axis.set(title="Distribution of Absolute Prediction Errors",
             xlabel="Absolute prediction error", ylabel="Number of vehicles")
    axis.xaxis.set_major_formatter(formatter)
    axis.legend(frameon=True)
    figure.tight_layout()
    path = CHART_DIRECTORY / "03_absolute_error_distribution.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    chart_paths.append(path)

    # Dashboard: one slide-friendly image summarizing the main evidence.
    within_2500 = 100.0 * float(np.mean(absolute_errors <= 2500.0))
    within_5000 = 100.0 * float(np.mean(absolute_errors <= 5000.0))
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    figure.suptitle("Used-Car Price Model: 600-Row Holdout Results",
                    fontsize=20, fontweight="bold", color=NAVY)

    axes[0, 0].scatter(actual, predicted, s=20, alpha=0.55, color=BLUE,
                       edgecolors="none")
    axes[0, 0].plot([low, high], [low, high], color=ORANGE, linewidth=1.8)
    axes[0, 0].set_title("Actual vs. predicted")
    axes[0, 0].set_xlabel("Actual price")
    axes[0, 0].set_ylabel("Predicted price")
    axes[0, 0].xaxis.set_major_formatter(formatter)
    axes[0, 0].yaxis.set_major_formatter(formatter)

    axes[0, 1].scatter(predicted, residuals, s=20, alpha=0.55, color=BLUE,
                       edgecolors="none")
    axes[0, 1].axhline(0.0, color=ORANGE, linewidth=1.8)
    axes[0, 1].set_title("Residual check")
    axes[0, 1].set_xlabel("Predicted price")
    axes[0, 1].set_ylabel("Predicted minus actual")
    axes[0, 1].xaxis.set_major_formatter(formatter)
    axes[0, 1].yaxis.set_major_formatter(formatter)

    axes[1, 0].hist(absolute_errors, bins=25, color=BLUE, alpha=0.85,
                    edgecolor="white")
    axes[1, 0].axvline(mae, color=ORANGE, linewidth=1.8)
    axes[1, 0].set_title("Absolute error distribution")
    axes[1, 0].set_xlabel("Absolute error")
    axes[1, 0].set_ylabel("Vehicles")
    axes[1, 0].xaxis.set_major_formatter(formatter)

    labels = ["Within\n$2,500", "Within\n$5,000"]
    percentages = [within_2500, within_5000]
    bars = axes[1, 1].bar(labels, percentages, color=[BLUE, NAVY], width=0.58)
    axes[1, 1].set_title("Share of predictions near actual price")
    axes[1, 1].set_ylabel("Percent of holdout vehicles")
    axes[1, 1].set_ylim(0.0, 100.0)
    for bar, percentage in zip(bars, percentages):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2.0,
                        percentage + 2.0, f"{percentage:.1f}%",
                        ha="center", fontweight="bold")
    axes[1, 1].text(
        0.5, 0.48,
        f"RMSE  ${rmse:,.0f}\nMAE     ${mae:,.0f}\nR²       {r_squared:.4f}",
        transform=axes[1, 1].transAxes, ha="center", va="center",
        fontsize=12, bbox={"facecolor": "white", "edgecolor": LIGHT_BLUE},
    )

    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    path = CHART_DIRECTORY / "00_presentation_dashboard.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    chart_paths.insert(0, path)
    return chart_paths


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
    chart_paths = save_presentation_charts(
        holdout_target, predictions, rmse, mae, r_squared
    )

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
    print("Presentation charts:")
    for path in chart_paths:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
