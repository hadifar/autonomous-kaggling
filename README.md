# Autonomous Kaggling

Applying autoresearch to a student health-condition prediction task: classify each record as `unhealthy`, `at-risk`, or `fit` from a mix of numeric and categorical features. The objective is the highest balanced accuracy possible.

## Evaluation

Submissions are scored on **balanced accuracy**: the mean of the per-class recalls, so each of the three classes contributes equally regardless of how often it appears in the data.

This matters because the classes are heavily imbalanced in the training set:

| Class | Count | Share |
| --- | --- | --- |
| `at-risk` | 592,561 | 85.9% |
| `unhealthy` | 57,724 | 8.4% |
| `fit` | 39,803 | 5.8% |

Because each class is weighted equally, predicting the majority class everywhere scores 0.3333 rather than 0.859. The two minority classes together account for roughly two thirds of the metric, so class weighting, stratified folds, and threshold tuning all carry real weight here.

## Submission File

For each `id` in the test set, predict one label for `health_condition`: `at-risk`, `unhealthy`, or `fit`. The file must include a header and follow this format:

```csv
id,health_condition
690088,at-risk
690089,at-risk
690090,at-risk
```

## Dataset Description

The raw dataset for this competition (both train and test) was inspired by the [College Student Health Behavior Dataset](https://www.kaggle.com/datasets/ziya07/college-student-health-behavior-dataset). Feature distributions are close to, but not exactly the same as, the original.

The files are located in the `./data/raw/` directory:

| File | Description |
| --- | --- |
| `train.csv` | Training set, with `health_condition` as the target |
| `test.csv` | Test set; predict `health_condition` for each row |
| `sample_submission.csv` | A sample submission in the correct format |

### Features

Both `train.csv` and `test.csv` share the same feature columns, keyed by `id`:

- **Sleep and vitals** — `sleep_duration`, `sleep_quality`, `heart_rate`, `bmi`
- **Activity** — `step_count`, `exercise_duration`, `calorie_expenditure`, `physical_activity_level`
- **Diet and habits** — `diet_type`, `water_intake`, `smoking_alcohol`, `stress_level`
- **Demographics** — `gender`

Some feature values are missing and will need to be handled during preprocessing.


## Baselines

Out-of-fold on the full training set (stratified 5-fold, 690,088 rows, seed 42):

| | Balanced accuracy |
| --- | --- |
| **Baseline** — majority class (`at-risk` everywhere) | 0.3333 |
| **XGBoost** — default hyperparameters | 0.8795 |
