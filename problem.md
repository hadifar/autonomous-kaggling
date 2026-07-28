# Student Health Risk

Everything in `src/train.py` is fair game: model choice, model architecture, optimizer,
hyperparameters, ensembling, training loop, model size. Work happens on a
dedicated experiment branch, one experiment at a time, and every run gets
logged — including the ones that fail. 


## Scope

| File | Status |
| --- | --- |
| `src/train.py` | **The file you modify.** Model, optimizer, training loop. |
| `src/data.py` | **Fixed — do not modify.** Loading, column roles, label encoding. |
| `src/config.py` | **Fixed — do not modify.** Paths, seed, fold count. |
| `README.md` | **Fixed — do not modify** Repository context. |
| `results.csv` | The experiment log. Append one row per run. |

`src/data.py` and `src/config.py` are the fixed evaluation harness. Holding them
constant is what makes two runs comparable — if they change, every number in
`results.csv` above the change becomes meaningless. Need a different encoding or
imputation? Do it inside `src/train.py`.

A run ends when the CV loop reports its OOF balanced accuracy — that score is
what gets logged and what decides whether the branch advances. Scoring is
out-of-fold only; Maximum allowed run-time is 60 minutes.

## Setup

Set up a new experiment together with the user:

1. **Agree on a run tag.** Propose one short meaningful tag (e.g., july28) with user ask. The branch `student-health-risk/<tag>`
   must not already exist — this is a fresh run, not a continuation.
   Check with `git rev-parse --verify student-health-risk/<tag>`; you want it to
   fail.
2. **Create the branch** from current master:

   ```bash
   git checkout -b student-health-risk/<tag>
   ```

3. **Read the in-scope files.** The repo is small, so read them in full for
   context: `README.md`, `src/data.py`, `src/config.py`, `src/train.py`.

4. **Initialize `results.csv`** with just the header row — the baseline gets
   recorded after the first run, not before; Baseline is xgboost with default params (`config/xgb_default_params.json`). 

5. **Confirm and go.** Confirm the setup looks good with the user. Once you have
   confirmation, kick off the experimentation.

Environment setup, if the clone is fresh: `uv sync` installs the locked
dependency set. Add dependencies to `pyproject.toml` and re-run `uv sync` —
never `pip install` into `.venv` by hand.

On a fresh clone:
`mkdir -p outputs/logs`

## Logging results

When an experiment is done, log it to `results.csv`. The file is
**tab-separated, not comma-separated** — commas break the descriptions.

Four columns, with a header row:

| Column | Meaning |
| --- | --- |
| `commit` | Short git commit hash, 7 characters |
| `balance_accuracy` | Balanced accuracy achieved; use `0.0000` for crashes |
| `status` | `keep`, `discard`, or `crash` |
| `description` | Short text description of what this experiment tried |

```
commit    balance_accuracy    status    description
a1b2c3d   0.9500              keep      baseline
b2c3d4e   0.9650              keep      seed ensembling
c3d4e5f   0.8000              discard   switch to svm classifier
d4e5f6a   0.0000              crash     random-forest with too many forests (OOM)
```

A run that isn't logged didn't happen. Log the failures with the same care as
the wins — the log is the record of what has already been ruled out.

## The experiment loop

The experiment runs on its dedicated `student-health-risk/<tag>` branch. 

History on this branch is **linear and append-only**: every experiment gets
committed and every commit stays. A rejected idea is undone with `git revert`,
never `git reset --hard` — that keeps the hashes in `results.csv` resolvable,
so `git show <commit>` always shows what a discarded run actually tried.

LOOP FOREVER:
   1. Check git state, and read `results.csv` for what's been tried and what
      scored best. `git log --oneline -12` if you need the code history.
   2. Tune `train.py` with one experimental idea - a run that changes both the
      features and the estimator can't tell you which one moved the score.
   3. Commit it and grab the hash:
      `git commit -am "<idea>" && EXP=$(git rev-parse --short=7 HEAD)`
   4. Run from the repo root:
      `uv run python src/train.py > outputs/logs/<tag>.log 2>&1`
      (redirect everything — do NOT use tee or let output flood your context.)
   5. Read the score: `grep -i "balanced accuracy" outputs/logs/<tag>.log`
   6. Empty grep means a crash. Read the trace with
      `tail -n 50 outputs/logs/<tag>.log` and try to fix it. After a few failed
      attempts, give up: log `crash` / `0.0000`, revert as in step 9, move on.
      A crash never reaches step 8, so it must be reverted here.
   7. Log the row to `results.csv` using `$EXP` as the commit (never commit
      this file — it is gitignored).
   8. Better than the best `keep` row so far? Log `keep`, leave the commit.
   9. Equal or worse? Log `discard`, then `git revert --no-edit "$EXP"`.

The idea is that you are a completely autonomous data scietist trying things out to win a kaggle competition. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate and get higher score. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.