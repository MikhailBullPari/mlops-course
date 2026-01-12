import pytest
import pandas as pd
from src.features import add_time_features


def test_add_time_features():
    df = pd.DataFrame(
        {
            "pickup_datetime": ["2021-01-01 10:30:00", "2021-01-02 15:45:00"],
            "fare_amount": [10.0, 15.0],
        }
    )
    result = add_time_features(df)

    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "pickup_datetime" not in result.columns
    assert result["hour"].tolist() == [10, 15]
    assert result["day_of_week"].tolist() == [4, 5]


def test_add_time_features_missing_column():
    df = pd.DataFrame({"fare_amount": [10.0, 15.0]})
    with pytest.raises(ValueError, match="Column 'pickup_datetime' not found"):
        add_time_features(df)


def test_add_time_features_invalid_datetime():
    df = pd.DataFrame({"pickup_datetime": ["invalid_date"], "fare_amount": [10.0]})
    with pytest.raises(ValueError, match="Error converting"):
        add_time_features(df)


def test_add_time_features_no_mutation():
    df = pd.DataFrame({"pickup_datetime": ["2021-01-01 10:30:00"], "fare_amount": [10.0]})
    original_columns = df.columns.tolist()
    add_time_features(df)

    assert df.columns.tolist() == original_columns
