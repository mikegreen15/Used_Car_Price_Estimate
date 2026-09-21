#!/usr/bin/env python3
"""Leakage-safe linear regression for the used-car price competition.

Only NumPy is used for array arithmetic.  The regression optimizer (Adam),
cleaning, encoding, feature construction, and prediction checks are implemented
in this file.  No machine-learning/statistics package, least-squares solver,
matrix inverse, pseudoinverse, or polynomial-fit helper is used.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CHANGE ONLY THESE PATHS FOR THE INSTRUCTOR'S FILES.
# ---------------------------------------------------------------------------
TRAIN_PATH = Path("car_price_train.csv")
TEST_PATH = Path("car_price_test.csv")
OUTPUT_PATH = Path("predictions.csv")


ID_COLUMN = "listing_id"
TARGET_COLUMN = "sale_price_usd"

# This field is deliberately unavailable at the stated prediction moment:
# predictions are made before negotiation and financing.  Keeping it would
# leak post-inspection financing information into the model.
EXCLUDED_POST_OUTCOME_COLUMNS = {"loan_approval_value_usd"}

NUMERIC_COLUMNS = [
    "model_year",
    "vehicle_age_years",
    "odometer_km",
    "annual_mileage_km",
    "engine_liters",
    "horsepower",
    "condition_score",
    "service_history_score",
    "prior_owners",
    "accident_count",
]

CATEGORICAL_COLUMNS = [
    "brand",
    "body_style",
    "fuel_type",
    "transmission",
    "region",
    "seller_type",
]

MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "unknown"}
LOWER_QUANTILE = 0.005
UPPER_QUANTILE = 0.995
DEFAULT_L2_STRENGTH = 0.003
LEARNING_RATE = 0.02
TRAINING_STEPS = 2200
GRADIENT_TOLERANCE = 2.5e-4
MINIMUM_STEPS = 200
PRICE_FLOOR = 1200.0
VALIDATION_ROWS = 600
VALIDATION_SEED = 20260921


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV without type guessing and normalize column names."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        original = list(reader.fieldnames)
        normalized = [name.strip() for name in original]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"CSV has duplicate column names after trimming: {path}")
        rows = []
        for source_row in reader:
            rows.append({new: (source_row.get(old) or "").strip()
                         for old, new in zip(original, normalized)})
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows, normalized


def parse_number(value: str) -> float:
    """Convert common numeric CSV text to float; invalid values become NaN."""
    text = value.strip()
    if text.lower() in MISSING_STRINGS:
        return math.nan
    try:
        number = float(text.replace(",", "").replace("$", ""))
    except ValueError:
        return math.nan
    return number if math.isfinite(number) else math.nan


def normalize_category(value: str) -> str:
    text = value.strip().lower()
    return "<missing>" if text in MISSING_STRINGS else text


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


class FeatureBuilder:
    """Training-only cleaning statistics and deterministic feature encoding."""

    def __init__(self) -> None:
        self.medians: dict[str, float] = {}
        self.bounds: dict[str, tuple[float, float]] = {}
        self.levels: dict[str, list[str]] = {}
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None

    def fit(self, rows: list[dict[str, str]]) -> None:
        for column in NUMERIC_COLUMNS:
            observed = np.asarray(
                [parse_number(row[column]) for row in rows], dtype=np.float64
            )
            finite = observed[np.isfinite(observed)]
            if finite.size == 0:
                raise ValueError(f"Training column has no finite values: {column}")
            self.medians[column] = float(np.median(finite))
            low, high = np.quantile(finite, [LOWER_QUANTILE, UPPER_QUANTILE])
            self.bounds[column] = (float(low), float(high))

        for column in CATEGORICAL_COLUMNS:
            self.levels[column] = sorted(
                {normalize_category(row[column]) for row in rows}
            )

        raw = np.asarray([self._raw_features(row) for row in rows], dtype=np.float64)
        self.means = raw.mean(axis=0)
        self.scales = raw.std(axis=0)
        self.scales[self.scales < 1.0e-12] = 1.0

    def _raw_features(self, row: dict[str, str]) -> list[float]:
        values: dict[str, float] = {}
        features: list[float] = []

        for column in NUMERIC_COLUMNS:
            value = parse_number(row[column])
            missing = not math.isfinite(value)
            if missing:
                value = self.medians[column]
            low, high = self.bounds[column]
            value = min(max(value, low), high)
            values[column] = value
            features.extend((value, 1.0 if missing else 0.0))

        # Fixed nonlinear basis terms keep the model linear in its coefficients.
        age = values["vehicle_age_years"]
        odometer = max(0.0, values["odometer_km"])
        horsepower = values["horsepower"]
        engine = values["engine_liters"]
        condition = values["condition_score"]
        service = values["service_history_score"]
        features.extend(
            (
                age * age,
                math.log1p(odometer),
                math.sqrt(odometer),
                horsepower * horsepower,
                engine * engine,
                condition * condition,
                service * service,
            )
        )

        # Drop the first sorted level as the reference category.  A category
        # seen only at test time maps to all-zero indicators, a safe fallback.
        for column in CATEGORICAL_COLUMNS:
            value = normalize_category(row[column])
            features.extend(1.0 if value == level else 0.0
                            for level in self.levels[column][1:])
        return features

    def transform(self, rows: list[dict[str, str]]) -> np.ndarray:
        if self.means is None or self.scales is None:
            raise RuntimeError("FeatureBuilder must be fitted before transform")
        raw = np.asarray([self._raw_features(row) for row in rows], dtype=np.float64)
        standardized = (raw - self.means) / self.scales
        return np.column_stack((np.ones(len(rows), dtype=np.float64), standardized))


class ManualLinearRegressor:
    """Linear regression with optional L2 penalty, trained by hand-coded Adam."""

    def __init__(self, l2_strength: float = DEFAULT_L2_STRENGTH) -> None:
        if not math.isfinite(l2_strength) or l2_strength < 0.0:
            raise ValueError("l2_strength must be a finite number greater than or equal to zero")
        self.l2_strength = float(l2_strength)
        self.weights: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.iterations_used = 0
        self.final_max_gradient = math.inf
        self.final_objective = math.inf
        self.converged = False

    def _objective_and_gradient(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return MSE + L2 objective and gradient; never penalize weights[0]."""
        residual = x @ weights - y
        mse = float(np.mean(residual**2))
        penalty = self.l2_strength * float(np.sum(weights[1:] ** 2))
        gradient = (2.0 / len(y)) * (x.T @ residual)
        gradient[1:] += 2.0 * self.l2_strength * weights[1:]
        return mse + penalty, gradient

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ManualLinearRegressor":
        """Fit with the required model.fit(X_train, y_train) interface."""
        x = np.asarray(X_train, dtype=np.float64)
        target = np.asarray(y_train, dtype=np.float64)
        if x.ndim != 2 or target.ndim != 1 or len(x) != len(target):
            raise ValueError("X_train must be 2D and y_train must be a matching 1D array")
        if len(target) < 2 or not np.isfinite(x).all() or not np.isfinite(target).all():
            raise ValueError("Training arrays must contain at least two finite observations")
        self.target_mean = float(target.mean())
        self.target_scale = float(target.std())
        if self.target_scale < 1.0e-12:
            self.target_scale = 1.0
        y = (target - self.target_mean) / self.target_scale

        weights = np.zeros(x.shape[1], dtype=np.float64)
        first_moment = np.zeros_like(weights)
        second_moment = np.zeros_like(weights)
        beta1, beta2 = 0.9, 0.999

        # Full-batch gradient descent is deterministic.  This is an explicitly
        # implemented optimizer, not a regression/optimization library call.
        for step in range(1, TRAINING_STEPS + 1):
            # Objective:
            #   mean((prediction - y)^2) + l2_strength * sum(weights[1:]^2)
            # weights[0] is the intercept, so it is deliberately not penalized.
            # At l2_strength=0 this is ordinary MSE and its ordinary gradient.
            self.final_objective, gradient = self._objective_and_gradient(x, y, weights)

            # Numerical stopping rule: after a short minimum run, stop when
            # every component of the regularized gradient is within tolerance.
            self.final_max_gradient = float(np.max(np.abs(gradient)))
            self.iterations_used = step
            if step >= MINIMUM_STEPS and self.final_max_gradient <= GRADIENT_TOLERANCE:
                self.converged = True
                break

            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
            correction = math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
            weights -= (
                LEARNING_RATE
                * correction
                * first_moment
                / (np.sqrt(second_moment) + 1.0e-8)
            )
        self.weights = weights
        self.intercept_ = self.target_mean + self.target_scale * weights[0]
        self.coef_ = self.target_scale * weights[1:].copy()
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return predictions with the required model.predict(X_test) interface."""
        if self.weights is None:
            raise RuntimeError("Model must be fitted before prediction")
        x = np.asarray(X_test, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.weights) or not np.isfinite(x).all():
            raise ValueError("X_test must be a finite 2D array with the fitted feature count")
        return self.target_mean + self.target_scale * (x @ self.weights)


def validate_ids(rows: list[dict[str, str]], label: str) -> None:
    ids = [row[ID_COLUMN].strip() for row in rows]
    if any(not value for value in ids):
        raise ValueError(f"{label} contains a blank {ID_COLUMN}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate {ID_COLUMN} values")


def write_predictions(path: Path, test_rows: list[dict[str, str]], values: np.ndarray) -> None:
    if len(values) != len(test_rows):
        raise RuntimeError("Prediction count does not match test row count")
    if not np.isfinite(values).all():
        raise RuntimeError("At least one prediction is not finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([ID_COLUMN, "sale_price_usd_pred"])
        for row, value in zip(test_rows, values):
            writer.writerow([row[ID_COLUMN], f"{float(value):.6f}"])


def print_validation_report(train_rows: list[dict[str, str]], target: np.ndarray) -> None:
    """Run a reproducible holdout evaluation using labels from the training CSV."""
    if len(train_rows) <= VALIDATION_ROWS:
        print("Validation report skipped: not enough labeled rows for the configured holdout.")
        return

    rng = np.random.default_rng(VALIDATION_SEED)
    shuffled = rng.permutation(len(train_rows))
    validation_indices = shuffled[:VALIDATION_ROWS]
    fitting_indices = shuffled[VALIDATION_ROWS:]
    fitting_rows = [train_rows[int(index)] for index in fitting_indices]
    validation_rows = [train_rows[int(index)] for index in validation_indices]
    fitting_target = target[fitting_indices]
    validation_target = target[validation_indices]

    # Learn every preprocessing value from the fitting rows only. This keeps
    # validation genuinely unseen during fitting and avoids preprocessing leak.
    builder = FeatureBuilder()
    builder.fit(fitting_rows)
    fitting_x = builder.transform(fitting_rows)
    validation_x = builder.transform(validation_rows)
    uncensored = fitting_target > PRICE_FLOOR + 1.0e-9

    validation_model = ManualLinearRegressor(l2_strength=DEFAULT_L2_STRENGTH)
    validation_model.fit(fitting_x[uncensored], fitting_target[uncensored])
    predictions = np.maximum(PRICE_FLOOR, validation_model.predict(validation_x))

    errors = predictions - validation_target
    absolute_errors = np.abs(errors)
    rmse = math.sqrt(float(np.mean(errors**2)))
    mae = float(np.mean(absolute_errors))
    median_absolute_error = float(np.median(absolute_errors))
    total_variation = float(np.sum((validation_target - validation_target.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(errors**2)) / total_variation
    within_2500 = 100.0 * float(np.mean(absolute_errors <= 2500.0))
    within_5000 = 100.0 * float(np.mean(absolute_errors <= 5000.0))

    print("\n--- Reproducible validation accuracy ---")
    print("These statistics use held-out training labels, not unknown test labels.")
    print(f"Split seed: {VALIDATION_SEED}")
    print(f"Fitting rows: {len(fitting_rows)}")
    print(f"Held-out validation rows: {len(validation_rows)}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"Median absolute error: ${median_absolute_error:,.2f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Predictions within $2,500: {within_2500:.1f}%")
    print(f"Predictions within $5,000: {within_5000:.1f}%")
    print(f"Actual price range: ${validation_target.min():,.2f} to ${validation_target.max():,.2f}")
    print(f"Predicted price range: ${predictions.min():,.2f} to ${predictions.max():,.2f}")
    print(f"Validation iterations used: {validation_model.iterations_used}")
    print(
        "Validation tolerance reached: "
        f"{'yes' if validation_model.converged else 'no; maximum iterations reached'}"
    )


def main() -> None:
    train_rows, train_columns = read_csv_rows(TRAIN_PATH)
    test_rows, test_columns = read_csv_rows(TEST_PATH)
    require_columns(
        train_columns,
        [ID_COLUMN, TARGET_COLUMN] + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS,
        "Training CSV",
    )
    require_columns(
        test_columns,
        [ID_COLUMN] + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS,
        "Test CSV",
    )
    validate_ids(train_rows, "Training CSV")
    validate_ids(test_rows, "Test CSV")

    target = np.asarray([parse_number(row[TARGET_COLUMN]) for row in train_rows])
    if not np.isfinite(target).all():
        raise ValueError(f"Training {TARGET_COLUMN} contains missing/non-finite values")
    if np.any(target <= 0.0):
        raise ValueError(f"Training {TARGET_COLUMN} must be positive")

    # Print demo-style accuracy before refitting on all labeled rows. The final
    # instructor test file has no target, so its true accuracy is unknowable.
    print_validation_report(train_rows, target)

    # Prices at the repeated data floor are censored observations: their latent
    # values are at or below the floor, not ordinary $1,200 measurements.  Fit
    # the linear relationship on uncensored observations and apply the known
    # floor after prediction.
    uncensored = target > PRICE_FLOOR + 1.0e-9
    if int(uncensored.sum()) < 2:
        raise ValueError("Not enough uncensored training rows to fit the model")

    builder = FeatureBuilder()
    builder.fit(train_rows)
    train_x = builder.transform(train_rows)[uncensored]
    test_x = builder.transform(test_rows)

    model = ManualLinearRegressor(l2_strength=DEFAULT_L2_STRENGTH)
    model.fit(train_x, target[uncensored])
    predictions = np.maximum(PRICE_FLOOR, model.predict(test_x))
    write_predictions(OUTPUT_PATH, test_rows, predictions)
    print("\n--- Final full-data fit and prediction file ---")
    print(f"Training rows used for final feature fitting: {len(train_rows)}")
    print(f"Uncensored target rows used for regression fitting: {int(uncensored.sum())}")
    print(f"Test rows predicted: {len(test_rows)}")
    print(f"Constructed feature columns including intercept: {train_x.shape[1]}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Maximum iterations: {TRAINING_STEPS}")
    print(f"Gradient stop tolerance: {GRADIENT_TOLERANCE:.1e}")
    print(f"L2 strength: {model.l2_strength}")
    print(f"Iterations used: {model.iterations_used}")
    print(f"Final maximum absolute gradient: {model.final_max_gradient:.3e}")
    print(f"Tolerance reached: {'yes' if model.converged else 'no; maximum iterations reached'}")
    print(f"Wrote {len(predictions)} predictions to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
