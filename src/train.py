"""XGBoost with masked-copy augmentation of fully-complete rows.

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
DECISIVE_WEIGHT = 3.0

AUGMENT_MULT = 2
BOUNDARY_BOOST = 1.0
HARDCLASS_BOOST = 1.0


def _pool_row_weights(
    base: pd.DataFrame,
    y_pool: np.ndarray,
    boundary_boost: float,
    hardclass_boost: float,
) -> np.ndarray:
    """Sampling weight per pool row, normalised.

    #1 boundary rows: `sleep_duration` within 0.5 of the rule's 6.0 / 7.0 cuts,
       where masking the field flips the branch rather than leaving it inferable.
    #2 minority-class rows: `fit` / `unhealthy`, which balanced accuracy weights
       far more heavily than the majority class.
    """
    w = np.ones(len(base))

    if boundary_boost != 1.0:
        sleep = base["sleep_duration"].to_numpy()
        near = np.minimum(np.abs(sleep - 6.0), np.abs(sleep - 7.0))
        w = w * np.where(near <= 0.5, boundary_boost, 1.0)

    if hardclass_boost != 1.0:
        w = w * np.where(y_pool != 0, hardclass_boost, 1.0)

    return w / w.sum()


def augment_rows(
    df: pd.DataFrame,
    seed: int,
    mult: int = 1,
    y: np.ndarray | None = None,
    boundary_boost: float = 1.0,
    hardclass_boost: float = 1.0,
):
    """Masked copies of fully-complete rows (variant b), optionally emphasised.

    Pool = rows with every feature present. Baseline (boosts == 1.0) emits `mult`
    independently-masked copies per pool row, k fields dropped per the natural
    missing-count distribution, decisive fields `DECISIVE_WEIGHT` over-weighted,
    stacked in pool-row-ordered blocks so the caller can tile the pool's labels.

    With a boost > 1 (needs `y`), the same total budget `mult * |pool|` is
    redistributed by sampling rows with replacement per `_pool_row_weights`.
    Sampling breaks tile alignment, so the aligned label vector is returned too.
    """
    rng = np.random.default_rng(seed)
    feat_cols = [c for c in df.columns if c != "id"]
    key_idx = [feat_cols.index(k) for k in DECISIVE]

    field_w = np.ones(len(feat_cols))
    field_w[key_idx] = DECISIVE_WEIGHT
    field_w = field_w / field_w.sum()

    nmiss = df[feat_cols].isna().sum(axis=1)
    cond = nmiss[nmiss >= 1].value_counts(normalize=True).sort_index()

    complete_mask = df[feat_cols].notna().all(axis=1).to_numpy()
    base = df[complete_mask].copy().reset_index(drop=True)
    emphasise = (boundary_boost != 1.0 or hardclass_boost != 1.0) and y is not None

    def _mask_block(pool):
        pool = pool.copy()
        n = len(pool)
        kcount = rng.choice(cond.index.to_numpy(), size=n, p=cond.to_numpy())
        logw = np.log(field_w)[None, :]
        gumbel = -np.log(-np.log(rng.random((n, len(feat_cols)))))
        rank = (logw + gumbel).argsort(axis=1)[:, ::-1].argsort(axis=1)
        drop_mask = rank < kcount[:, None]
        for j, col in enumerate(feat_cols):
            pool.loc[drop_mask[:, j], col] = np.nan
        return pool

    if not emphasise:
        # Original path: mult copies of every pool row, block order preserved.
        out = pd.concat([_mask_block(base) for _ in range(mult)], ignore_index=True)
        return out

    # Emphasised path: same total budget, sampled by row weight (with replacement).
    y_pool = y[complete_mask]
    p = _pool_row_weights(base, y_pool, boundary_boost, hardclass_boost)
    total = mult * len(base)
    idx = rng.choice(len(base), size=total, replace=True, p=p)
    sampled = base.iloc[idx].reset_index(drop=True)
    masked = _mask_block(sampled)
    return masked, y_pool[idx]


def main() -> None:
    data = load_train()
    x, y = as_categorical(data["x"]), data["y"]

    print(f"{len(x):,} rows, {x.shape[1]} features, {len(CLASSES)} classes")

    oof = pd.Series(index=x.index, dtype=float)
    folds = StratifiedKFold(
        n_splits=settings.n_splits, shuffle=True, random_state=settings.seed
    )

    for fold, (train_idx, val_idx) in enumerate(folds.split(x, y), start=1):
        xtr, ytr = x.iloc[train_idx], y.iloc[train_idx].to_numpy()

        # Augment training folds only: masked copies of validation rows would
        # leak their labels into the OOF score.
        extra = augment_rows(
            xtr, seed=settings.seed + fold, mult=AUGMENT_MULT, y=ytr,
            boundary_boost=BOUNDARY_BOOST, hardclass_boost=HARDCLASS_BOOST,
        )
        if isinstance(extra, tuple):
            x_extra, y_extra = extra
        else:
            complete = xtr.notna().all(axis=1).to_numpy()
            x_extra, y_extra = extra, np.tile(ytr[complete], AUGMENT_MULT)

        x_fit = pd.concat([xtr.reset_index(drop=True), x_extra], ignore_index=True)
        y_fit = pd.Series(np.concatenate([ytr, y_extra]))

        model = build_model()
        weights = compute_sample_weight("balanced", y_fit)
        model.fit(x_fit, y_fit, sample_weight=weights)
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
