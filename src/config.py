"""Project settings.

Paths and run-level knobs, overridable with `SHR_`-prefixed environment
variables (e.g. `SHR_SEED=7`, `SHR_DATA_DIR=/mnt/data`).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHR_", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data" / "raw"

    seed: int = 42
    n_splits: int = 5

    @property
    def train_csv(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_dir / "test.csv"

    @property
    def sample_submission_csv(self) -> Path:
        return self.data_dir / "sample_submission.csv"


settings = Settings()
