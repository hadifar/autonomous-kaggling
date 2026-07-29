"""XGBoost + a specialist model for rule-undecidable rows.

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


def undecidable(x: pd.DataFrame) -> np.ndarray:
    """Rows where the label rule cannot be evaluated because a feature it needs
    is missing. The rule (verified on the clean source data) is:

        sleep_duration >= 7 and stress=low and activity=active -> fit
        sleep_duration <  6 and stress=high                    -> unhealthy
        otherwise                                              -> at-risk
    """
    sd = x["sleep_duration"]
    st = x["stress_level"].astype(object)
    pa = x["physical_activity_level"].astype(object)

    certain_fit = (sd >= 7) & (st == "low") & (pa == "active") & sd.notna() & st.notna() & pa.notna()
    certain_unhealthy = (sd < 6) & (st == "high") & sd.notna() & st.notna()
    not_fit = ((sd < 7) & sd.notna()) | ((st != "low") & st.notna()) | ((pa != "active") & pa.notna())
    not_unhealthy = ((sd >= 6) & sd.notna()) | ((st != "high") & st.notna())

    return ~(certain_fit | certain_unhealthy | (not_fit & not_unhealthy)).values


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
        xva = x.iloc[val_idx]

        model = build_model()
        model.fit(xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        predictions = model.predict(xva)

        # Rows the rule cannot decide carry all the remaining error; give them a
        # model that spends its whole capacity on that subpopulation.
        u_tr, u_va = undecidable(xtr), undecidable(xva)
        specialist = build_model()
        specialist.fit(
            xtr[u_tr], ytr[u_tr],
            sample_weight=compute_sample_weight("balanced", ytr[u_tr]),
        )
        predictions[u_va] = specialist.predict(xva[u_va])

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
