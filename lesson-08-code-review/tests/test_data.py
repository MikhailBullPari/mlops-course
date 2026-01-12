import pytest
import pandas as pd
from src.data import load_data, split_data, validate_data


def test_load_data():
    df = load_data("data/uber.csv")
    assert not df.empty
    assert "fare_amount" in df.columns
    assert "pickup_datetime" in df.columns


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")


def test_validate_data_missing_columns():
    df = pd.DataFrame({"col1": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df)


def test_validate_data_empty():
    df = pd.DataFrame(columns=["fare_amount", "pickup_datetime"])
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_data(df)


def test_clean_data_with_nulls():
    from src.data import clean_data

    df = pd.DataFrame(
        {
            "fare_amount": [10.0, None, 15.0],
            "pickup_datetime": ["2021-01-01", "2021-01-02", "2021-01-03"],
        }
    )
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert cleaned["fare_amount"].isnull().sum() == 0


def test_clean_data_with_negative_fare():
    from src.data import clean_data

    df = pd.DataFrame(
        {
            "fare_amount": [10.0, -5.0, 15.0],
            "pickup_datetime": ["2021-01-01", "2021-01-02", "2021-01-03"],
        }
    )
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert (cleaned["fare_amount"] > 0).all()


def test_split_data():
    df = pd.DataFrame(
        {
            "fare_amount": [10.0, 15.0, 20.0, 25.0, 30.0],
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
        }
    )
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1
    assert "fare_amount" not in X_train.columns
    assert "fare_amount" not in X_test.columns


def test_split_data_reproducibility():
    df = pd.DataFrame({"fare_amount": [10.0, 15.0, 20.0, 25.0, 30.0], "feature1": [1, 2, 3, 4, 5]})
    X_train1, X_test1, y_train1, y_test1 = split_data(df, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(df, random_state=42)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
