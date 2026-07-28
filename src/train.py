"""CatBoost with balanced class weights.

Balanced accuracy weights each class equally, but 86% of the training rows are
`at-risk`. Inverse-frequency sample weights align the training objective with
the metric.

CatBoost's ordered boosting and target-statistic categorical handling make a
genuinely different fit than the XGBoost/LightGBM histogram split search.
"""

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from config import settings
from data import CATEGORICAL, CLASSES, load_train


def build_model() -> CatBoostClassifier:
    """Single source of truth for the model configuration used by every CV
    fold."""
    return CatBoostClassifier(
        iterations=600,
        learning_rate=0.1,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        cat_features=CATEGORICAL,
        random_seed=settings.seed,
        thread_count=-1,
        verbose=0,
    )


def main() -> None:
    data = load_train()
    x, y = data["x"].copy(), data["y"]
    # CatBoost needs categorical columns as strings with no NaN.
    x[CATEGORICAL] = x[CATEGORICAL].fillna("missing").astype(str)

    print(f"{len(x):,} rows, {x.shape[1]} features, {len(CLASSES)} classes")

    oof = pd.Series(index=x.index, dtype=float)
    folds = StratifiedKFold(
        n_splits=settings.n_splits, shuffle=True, random_state=settings.seed
    )

    for fold, (train_idx, val_idx) in enumerate(folds.split(x, y), start=1):
        model = build_model()
        weights = compute_sample_weight("balanced", y.iloc[train_idx])
        model.fit(x.iloc[train_idx], y.iloc[train_idx], sample_weight=weights)
        # CatBoost returns an (n, 1) column vector rather than sklearn's 1-D array.
        predictions = model.predict(x.iloc[val_idx]).ravel()
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
