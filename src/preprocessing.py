import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace(",", "")

    if value == "":
        return np.nan

    return float(value)


def parse_volume(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace(",", "")

    if value == "" or value == "-":
        return np.nan

    multiplier = 1.0

    if value.endswith("K"):
        multiplier = 1_000
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000
        value = value[:-1]
    elif value.endswith("B"):
        multiplier = 1_000_000_000
        value = value[:-1]

    try:
        return float(value) * multiplier
    except ValueError:
        return np.nan


def parse_change_percent(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace("%", "").replace(",", "")

    if value == "" or value == "-":
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = df["Price"].apply(clean_numeric_value)
    df["Open"] = df["Open"].apply(clean_numeric_value)
    df["High"] = df["High"].apply(clean_numeric_value)
    df["Low"] = df["Low"].apply(clean_numeric_value)
    df["Vol."] = df["Vol."].apply(parse_volume)
    df["Change %"] = df["Change %"].apply(parse_change_percent)

    df = df.dropna(subset=["Date", "Price"])
    df = df.sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df


def time_split_dataframe(df: pd.DataFrame, train_ratio: float = 0.8):
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def scale_features_and_target(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols, target_col="Price"):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_scaled = feature_scaler.fit_transform(train_df[feature_cols])
    X_test_scaled = feature_scaler.transform(test_df[feature_cols])

    y_train_scaled = target_scaler.fit_transform(train_df[[target_col]])
    y_test_scaled = target_scaler.transform(test_df[[target_col]])

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler


def create_sequences(features, target, window_size=60, horizon=1):
    X, y = [], []

    for i in range(window_size, len(features) - horizon + 1):
        X.append(features[i - window_size:i])
        y.append(target[i:i + horizon].flatten())

    return np.array(X), np.array(y)