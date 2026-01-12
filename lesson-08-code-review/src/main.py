import os
from pathlib import Path
from src.data import load_data, split_data
from src.features import add_time_features
from src.model import TaxiFareModel


DATA_PATH = os.getenv("DATA_PATH", "data/uber.csv")
MODEL_PATH = "models/taxi_fare_model.pkl"
METRICS_PATH = "models/metrics.json"


def main() -> None:
    raw_data = load_data(DATA_PATH)
    print(f"Data loaded {len(raw_data)} rows")
    processed_data = add_time_features(raw_data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("Training model...")
    model = TaxiFareModel(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print(f"R2 Score: {metrics['r2_score']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")

    Path("models").mkdir(exist_ok=True)
    model.save(MODEL_PATH, METRICS_PATH)
    print(f"Model saved {MODEL_PATH}")
    print(f"Metrics saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
