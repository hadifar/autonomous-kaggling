"""XGBoost with masked-copy augmentation of fully-observed rows.

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


DECISIVE = ["sleep_duration", "stress_level", "physical_activity_level"]
"""The three columns the label rule reads. Masking these is what generates the
hard cases the model has to hedge on."""

DECISIVE_MULTIPLIER = 3.0
AUGMENT_COPIES = 2

def augment(
    x: pd.DataFrame, y: pd.Series, rates: dict[str, float], seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Append masked copies of the fully-observed rows.

    Each copy keeps its label but has fields blanked at the column's natural
    missingness rate, with the decisive columns masked `DECISIVE_MULTIPLIER`x
    more often. The intent is extra supervision on rows that look undecidable
    but whose true label is known.
    """
    rng = np.random.default_rng(seed)
    full = np.where(x.notna().all(axis=1).values)[0]
    idx = np.concatenate([full] * AUGMENT_COPIES)

    extra = x.iloc[idx].copy()
    for column in extra.columns:
        rate = rates[column] * (DECISIVE_MULTIPLIER if column in DECISIVE else 1.0)
        masked = rng.random(len(extra)) < min(rate, 0.95)
        if masked.any():
            extra[column] = extra[column].where(~masked)

    return pd.concat([x, extra], axis=0), pd.concat([y, y.iloc[idx]], axis=0)


def main() -> None:
    data = load_train()
    x, y = as_categorical(data["x"]), data["y"]

    print(f"{len(x):,} rows, {x.shape[1]} features, {len(CLASSES)} classes")
    rates = {c: float(x[c].isna().mean()) for c in x.columns}

    oof = pd.Series(index=x.index, dtype=float)
    folds = StratifiedKFold(
        n_splits=settings.n_splits, shuffle=True, random_state=settings.seed
    )

    for fold, (train_idx, val_idx) in enumerate(folds.split(x, y), start=1):
        # Augment the training folds only -- masked copies of validation rows
        # would leak their labels into the OOF score.
        xtr, ytr = augment(x.iloc[train_idx], y.iloc[train_idx], rates, settings.seed + fold)

        model = build_model()
        model.fit(xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        predictions = model.predict(x.iloc[val_idx])
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
