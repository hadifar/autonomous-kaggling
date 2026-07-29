"""Weighted XGBoost + LightGBM + CatBoost blend with balanced class weights.

Balanced accuracy weights each class equally, but 86% of the training rows are
`at-risk`. Inverse-frequency sample weights align the training objective with
the metric.

Blend weights come from a 60-trial Optuna search over the simplex, tuned on a
calibration split and scored on a held-out third split (0.95133 there against
0.95118 for XGBoost alone).
"""

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import settings
from data import CATEGORICAL, CLASSES, as_categorical, load_train

BLEND_WEIGHTS = (0.502, 0.400, 0.098)


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


def build_lightgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=settings.seed,
        n_jobs=-1,
        verbose=-1,
    )


def build_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=400,
        learning_rate=0.1,
        depth=8,
        loss_function="MultiClass",
        cat_features=CATEGORICAL,
        random_seed=settings.seed,
        thread_count=-1,
        verbose=0,
    )


def for_catboost(x: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants categorical columns as strings with no NaN."""
    x = x.copy()
    x[CATEGORICAL] = x[CATEGORICAL].fillna("missing").astype(str)
    return x


def main() -> None:
    data = load_train()
    x, y = as_categorical(data["x"]), data["y"]
    x_cat = for_catboost(data["x"])

    print(f"{len(x):,} rows, {x.shape[1]} features, {len(CLASSES)} classes")

    oof = pd.Series(index=x.index, dtype=float)
    folds = StratifiedKFold(
        n_splits=settings.n_splits, shuffle=True, random_state=settings.seed
    )

    for fold, (train_idx, val_idx) in enumerate(folds.split(x, y), start=1):
        y_train = y.iloc[train_idx]
        weights = compute_sample_weight("balanced", y_train)

        xgb = build_model()
        xgb.fit(x.iloc[train_idx], y_train, sample_weight=weights)

        lgbm = build_lightgbm()
        lgbm.fit(x.iloc[train_idx], y_train, sample_weight=weights)

        cat = build_catboost()
        cat.fit(x_cat.iloc[train_idx], y_train, sample_weight=weights)

        probabilities = (
            BLEND_WEIGHTS[0] * xgb.predict_proba(x.iloc[val_idx])
            + BLEND_WEIGHTS[1] * lgbm.predict_proba(x.iloc[val_idx])
            + BLEND_WEIGHTS[2] * cat.predict_proba(x_cat.iloc[val_idx])
        )
        predictions = probabilities.argmax(axis=1)
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
