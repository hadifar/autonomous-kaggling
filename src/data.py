"""Dataset loading and column roles.

Loading is model-agnostic: features come back with their raw dtypes and missing
values intact. Encoding choices — category dtype, one-hot, imputation — belong
to whichever model consumes them.
"""

from typing import Any

import pandas as pd

from config import settings

TARGET = "health_condition"
ID_COLUMN = "id"

CLASSES = [
    "at-risk",
    "fit",
    "unhealthy",
]
"""Label for integer code `i` is `CLASSES[i]`; alphabetical, matching the
encoding the baselines in `README.md` were scored under."""

CATEGORICAL = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

NUMERIC = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in (TARGET, ID_COLUMN) if c in df.columns])


def as_categorical(x: pd.DataFrame) -> pd.DataFrame:
    """Cast the categorical columns to pandas `category` dtype.

    For learners with native categorical support (XGBoost, LightGBM); other
    models want an encoder instead.
    """
    return x.astype(dict.fromkeys(CATEGORICAL, "category"))


def load_train() -> dict[str, Any]:
    """Keys: `x` (features), `y` (integer-encoded labels).

    Label `i` in `y` maps back to `CLASSES[i]`.
    """
    df = pd.read_csv(settings.train_csv)

    y = df[TARGET].map({label: i for i, label in enumerate(CLASSES)})
    return {"x": _features(df), "y": y}


def load_test() -> dict[str, Any]:
    """Keys: `x` (features), `ids` (the `id` column, for the submission)."""
    df = pd.read_csv(settings.test_csv)
    return {"x": _features(df), "ids": df[ID_COLUMN]}
