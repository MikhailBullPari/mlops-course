from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path)
        validate_data(data)
        data = clean_data(data)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")


def validate_data(data: pd.DataFrame) -> None:
    required_columns = ["fare_amount", "pickup_datetime"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("Dataset is empty")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(data)

    data = data.dropna()

    data = data[data["fare_amount"] > 0]

    removed_rows = initial_rows - len(data)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing or invalid values")

    if data.empty:
        raise ValueError("No valid data remaining after cleaning")

    return data


def split_data(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = data.drop("fare_amount", axis=1)
    target = data["fare_amount"]
    return train_test_split(features, target, test_size=test_size, random_state=random_state)
