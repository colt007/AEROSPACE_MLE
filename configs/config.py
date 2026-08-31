"""Project configuration loaded from environment variables with local defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _get_path(name: str, default: str) -> Path:
	path = Path(os.getenv(name, default))
	return path if path.is_absolute() else ROOT_DIR / path


def _get_int(name: str, default: int) -> int:
	return int(os.getenv(name, str(default)))


def _get_float(name: str, default: float) -> float:
	return float(os.getenv(name, str(default)))


WINDOW_SIZE = _get_int("WINDOW_SIZE", 30)
NUM_FEATURES = _get_int("NUM_FEATURES", 24)
HIDDEN_SIZE = _get_int("HIDDEN_SIZE", 128)
BATCH_SIZE = _get_int("BATCH_SIZE", 64)
LEARNING_RATE = _get_float("LEARNING_RATE", 1e-3)
WEIGHT_DECAY = _get_float("WEIGHT_DECAY", 1e-4)
EPOCHS = _get_int("EPOCHS", 15)
TRAIN_SPLIT = _get_float("TRAIN_SPLIT", 0.8)
RANDOM_SEED = _get_int("RANDOM_SEED", 42)

PROCESSED_PARQUET_PATH = _get_path(
	"PROCESSED_PARQUET_PATH", "datasets/processed/output_parquet.parquet"
)
STATS_PATH = _get_path("STATS_PATH", "datasets/processed/output_json.json")
TRAIN_DATA_PATH = _get_path("TRAIN_DATA_PATH", "datasets/raw/train_FD001.txt")
TEST_DATA_PATH = _get_path("TEST_DATA_PATH", "datasets/raw/test_FD001.txt")
RUL_DATA_PATH = _get_path("RUL_DATA_PATH", "datasets/raw/RUL_FD001.txt")
MODEL_OUTPUT_PATH = _get_path("MODEL_OUTPUT_PATH", "models/first_model.pt")
MODEL_URI = os.getenv(
	"MODEL_URI",
	"mlruns/1/models/m-2a69c4f085944505b4bd9b3efd3aa707/artifacts",
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "CMAPSS_BASELINE")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "34a55fbcd4b6426c98e9d24e673c0d35")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = _get_int("APP_PORT", 8000)
API_URL = os.getenv("API_URL", f"http://127.0.0.1:{APP_PORT}")