import os


RANDOM_STATE = 42

DATA_PATH = os.getenv("DATA_PATH", "data/uber.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

MODEL_OUTPUT_DIR = "models"
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "taxi_fare_model.pkl")
METRICS_PATH = os.path.join(MODEL_OUTPUT_DIR, "metrics.json")

MODEL_PARAMS = {
    "n_estimators": int(os.getenv("N_ESTIMATORS", "100")),
    "learning_rate": float(os.getenv("LEARNING_RATE", "0.1")),
    "max_depth": int(os.getenv("MAX_DEPTH", "3")),
    "random_state": RANDOM_STATE
}
