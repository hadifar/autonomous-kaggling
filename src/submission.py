"""Fit the winning configuration on all of train and predict the test set.

Training mirrors `train.py` exactly — same model, same balanced sample weights —
but fits on the full training set instead of holding out folds, since there is
no OOF score to compute here.

Artifacts land in `outputs/submissions/<tag>/`:

    submission.csv    the file Kaggle scores
    message.txt       the submission message, read back by ./submit.sh

The upload is a separate, explicit step: this script never talks to Kaggle.
Run `./submit.sh <tag>` when you want the submission to actually count against
your daily quota.

Usage:
    uv run python src/submission.py --tag shr-v1 --message "xgb + balanced weights"
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from config import PROJECT_ROOT, settings
from data import CLASSES, as_categorical, load_test, load_train
from train import build_model

def build_submission() -> pd.DataFrame:
    """Fit on the full training set and label the test rows."""
    train, test = load_train(), load_test()
    x, y = as_categorical(train["x"]), train["y"]

    print(f"training on {len(x):,} rows, {x.shape[1]} features")
    model = build_model()
    model.fit(x, y, sample_weight=compute_sample_weight("balanced", y))

    x_test = as_categorical(test["x"])
    print(f"predicting {len(x_test):,} test rows")
    predictions = model.predict(x_test)

    return pd.DataFrame(
        {
            "id": test["ids"],
            "health_condition": [CLASSES[i] for i in predictions],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", required=True, help="run tag; names the output subdirectory"
    )
    parser.add_argument(
        "--message", required=True, help="submission message shown on Kaggle"
    )
    args = parser.parse_args()

    submission = build_submission()

    out_dir = PROJECT_ROOT / "outputs" / "submissions" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "submission.csv"
    submission.to_csv(csv_path, index=False)

    (out_dir / "message.txt").write_text(args.message)

    print(f"\nwrote {csv_path} ({len(submission):,} rows)")
    print("\nclass distribution:")
    shares = submission["health_condition"].value_counts(normalize=True)
    for label in CLASSES:
        print(f"  {label:<10} {shares.get(label, 0.0):.4f}")

    print(f"\nto submit:  ./submit.sh {args.tag}")


if __name__ == "__main__":
    main()
