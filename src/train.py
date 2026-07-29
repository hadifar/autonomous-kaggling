"""XGBoost with balanced class weights, bagged over 10 seeds.

Balanced accuracy weights each class equally, but 86% of the training rows are
`at-risk`. Inverse-frequency sample weights align the training objective with
the metric.

Errors concentrate on ambiguous rows whose predicted probabilities sit near the
decision boundary, where the subsample/colsample draw decides the label. A
3-seed average was too small to separate from noise; 10 seeds averages that
draw away properly.
"""

import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import settings
from data import CLASSES, as_categorical, load_train


N_SEEDS = 10


def build_model(seed_offset: int = 0) -> XGBClassifier:
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
        random_state=settings.seed + seed_offset,
        n_jobs=-1,
    )


def main() -> None:
    data = load_train()
    x, y = as_categorical(data["x"]), data["y"]

    print(f"{len(x):,} rows, {x.shape[1]} features, {len(CLASSES)} classes")

    oof = pd.Series(index=x.index, dtype=float)
    folds = StratifiedKFold(
        n_splits=settings.n_splits, shuffle=True, random_state=settings.seed
    )

    for fold, (train_idx, val_idx) in enumerate(folds.split(x, y), start=1):
        weights = compute_sample_weight("balanced", y.iloc[train_idx])
        probabilities = None
        for seed_offset in range(N_SEEDS):
            model = build_model(seed_offset)
            model.fit(x.iloc[train_idx], y.iloc[train_idx], sample_weight=weights)
            seed_probabilities = model.predict_proba(x.iloc[val_idx])
            probabilities = (
                seed_probabilities
                if probabilities is None
                else probabilities + seed_probabilities
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
