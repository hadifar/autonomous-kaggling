"""XGBoost with latent rule-condition probabilities as extra features.

Balanced accuracy weights each class equally, but 86% of the training rows are
`at-risk`. Inverse-frequency sample weights align the training objective with
the metric.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import settings
from data import CLASSES, as_categorical, load_train


def build_model() -> XGBClassifier:
    """Single source of truth for the model configuration used by every CV
    fold."""
    return XGBClassifier(
        max_depth=6,
        learning_rate=0.09755452581197879,
        subsample=0.7917074280055386,
        colsample_bytree=0.7302102879528256,
        min_child_weight=19,
        reg_lambda=0.047675439361864844,
        reg_alpha=0.0020126791514167887,
        gamma=0.0036943104482137935,
        max_bin=1024,
        n_estimators=153,
        enable_categorical=True,
        tree_method="hist",
        random_state=settings.seed,
        n_jobs=-1,
    )


LATENT = {
    "s7": ("sleep_duration", lambda c: c >= 7),
    "s6": ("sleep_duration", lambda c: c < 6),
    "lo": ("stress_level", lambda c: c.astype(object) == "low"),
    "hi": ("stress_level", lambda c: c.astype(object) == "high"),
    "ac": ("physical_activity_level", lambda c: c.astype(object) == "active"),
}
"""The binary conditions the label rule is built from. Each is learnable from the
other columns on the rows where it is observed, which is far more supervision per
target than the 3-class label gives."""


def add_latent(x_fit: pd.DataFrame, y_unused, x_out: pd.DataFrame) -> pd.DataFrame:
    """Append P(condition) for each rule condition, fit on `x_fit`, applied to
    `x_out`. Where the underlying column is observed in `x_out`, the truth is
    used instead of the prediction."""
    others = [c for c in x_fit.columns if c not in
              ("sleep_duration", "stress_level", "physical_activity_level")]
    out = x_out.copy()
    for key, (col, cond) in LATENT.items():
        observed = x_fit[col].notna()
        model = XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=120, subsample=0.8,
            colsample_bytree=0.8, max_bin=256, enable_categorical=True,
            tree_method="hist", n_jobs=-1, random_state=settings.seed,
        )
        model.fit(x_fit.loc[observed, others], cond(x_fit[col][observed]).astype(int))
        p = model.predict_proba(x_out[others])[:, 1]
        seen = x_out[col].notna().values
        out["z_" + key] = np.where(seen, cond(x_out[col]).astype(float).values, p)
    return out


def main() -> None:
    data = load_train()
    x, y = as_categorical(data["x"]), data["y"]

    print(f"{len(x):,} rows, {x.shape[1]} features, {len(CLASSES)} classes")

    oof = pd.Series(index=x.index, dtype=float)
    folds = StratifiedKFold(
        n_splits=settings.n_splits, shuffle=True, random_state=settings.seed
    )

    for fold, (train_idx, val_idx) in enumerate(folds.split(x, y), start=1):
        xtr, ytr = x.iloc[train_idx], y.iloc[train_idx]

        model = build_model()
        weights = compute_sample_weight("balanced", ytr)
        model.fit(add_latent(xtr, ytr, xtr), ytr, sample_weight=weights)
        predictions = model.predict(add_latent(xtr, ytr, x.iloc[val_idx]))
        oof.iloc[val_idx] = predictions

        fold_score = balanced_accuracy_score(y.iloc[val_idx], predictions)
        print(f"  fold {fold}: {fold_score:.4f}")

    assert not oof.isna().any(), "some rows were never assigned an OOF prediction"
    oof = oof.astype(int)

    print(f"\nOOF balanced accuracy: {balanced_accuracy_score(y, oof):.4f}")

    print("\nPer-class recall:")
    for i, label in enumerate(CLASSES):
        mask = y == i
        recall = (oof[mask] == i).mean()
        print(f"  {label:<10} {recall:.4f}  (n={int(mask.sum()):,})")


if __name__ == "__main__":
    main()
