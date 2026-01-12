import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pickup_datetime" not in df.columns:
        raise ValueError("Column 'pickup_datetime' not found in DataFrame")

    try:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    except Exception as e:
        raise ValueError(f"Error converting 'pickup_datetime' to datetime: {str(e)}")

    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

    df = df.drop("pickup_datetime", axis=1)

    columns_to_drop = ["Unnamed: 0", "key"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    return df
