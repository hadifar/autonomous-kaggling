"""Build a leaderboard-targeted submission.

`submission.py` mirrors `train.py` exactly so the submitted model is the one the
OOF loop scored. This script optimises for the public score instead, which is a
different objective: it is a *single* draw on roughly 59k rows, so variance in
the predictions matters as much as the expected score.

Two changes follow from that:

* **Every model sees all 690,088 rows.** Training size has a measured monotone
  effect (60% -> 0.94974, 80% -> 0.95003, 100% -> 0.95022 on a fixed test set).
* **Average over seeds and model families.** On OOF these are neutral — the
  learners make correlated errors — but averaging cuts the seed-to-seed variance
  that decides a single leaderboard draw. Only 0.42% of test rows flip with the
  seed, and this pins those down rather than leaving them to one lucky draw.

Usage:
    uv run python src/submission_lb.py --tag lb-blend --message "..."
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from config import PROJECT_ROOT, settings
from data import CATEGORICAL, CLASSES, as_categorical, load_test, load_train
from train import build_model

N_SEEDS = 5
BLEND_WEIGHTS = {"xgb": 0.502, "lgbm": 0.400, "cat": 0.098}


def _for_catboost(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x[CATEGORICAL] = x[CATEGORICAL].fillna("missing").astype(str)
    return x


def _accumulate(total: np.ndarray | None, part: np.ndarray) -> np.ndarray:
    return part if total is None else total + part


def build_probabilities(x, y, x_test, x_cat, x_test_cat) -> np.ndarray:
    """Seed-averaged probabilities from each family, blended by BLEND_WEIGHTS."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    weights = compute_sample_weight("balanced", y)
    blended = None

    for name in ("xgb", "lgbm", "cat"):
        family = None
        for seed in range(N_SEEDS):
            if name == "xgb":
                model = build_model()
                model.set_params(random_state=settings.seed + seed)
                model.fit(x, y, sample_weight=weights)
                probabilities = model.predict_proba(x_test)
            elif name == "lgbm":
                model = LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=63,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    random_state=settings.seed + seed,
                    n_jobs=-1,
                    verbose=-1,
                )
                model.fit(x, y, sample_weight=weights)
                probabilities = model.predict_proba(x_test)
            else:
                model = CatBoostClassifier(
                    iterations=400,
                    learning_rate=0.1,
                    depth=8,
                    loss_function="MultiClass",
                    cat_features=CATEGORICAL,
                    random_seed=settings.seed + seed,
                    thread_count=-1,
                    verbose=0,
                )
                model.fit(x_cat, y, sample_weight=weights)
                probabilities = model.predict_proba(x_test_cat)

            family = _accumulate(family, probabilities)
            print(f"  {name} seed {seed + 1}/{N_SEEDS} done", flush=True)

        blended = _accumulate(blended, BLEND_WEIGHTS[name] * (family / N_SEEDS))

    return blended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    train, test = load_train(), load_test()
    x, y = as_categorical(train["x"]), train["y"]
    x_test = as_categorical(test["x"])
    x_cat, x_test_cat = _for_catboost(train["x"]), _for_catboost(test["x"])

    print(f"training on {len(x):,} rows (100%), {N_SEEDS} seeds x 3 families")
    probabilities = build_probabilities(x, y, x_test, x_cat, x_test_cat)
    predictions = probabilities.argmax(axis=1)

    submission = pd.DataFrame(
        {
            "id": test["ids"],
            "health_condition": [CLASSES[i] for i in predictions],
        }
    )

    out_dir = PROJECT_ROOT / "outputs" / "submissions" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_dir / "submission.csv", index=False)
    (out_dir / "message.txt").write_text(args.message)

    print(f"\nwrote {out_dir / 'submission.csv'} ({len(submission):,} rows)")
    shares = submission["health_condition"].value_counts(normalize=True)
    for label in CLASSES:
        print(f"  {label:<10} {shares.get(label, 0.0):.4f}")
    print(f"\nto submit:  ./submit.sh {args.tag}")


if __name__ == "__main__":
    main()
