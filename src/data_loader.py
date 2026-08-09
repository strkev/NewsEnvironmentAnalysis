import os

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def load_raw_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    print(f"Loading raw data from: {file_path}")
    df = pd.read_csv(file_path)
    df["full_text"] = df["Title"].fillna("") + " " + df["Article Text"].fillna("")
    return df


def save_processed_data(df: pd.DataFrame, file_path: str = PROCESSED_DATA_PATH):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Cached processed data to: {file_path}")


def load_processed_data(file_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"No cached data found '{file_path}'. Run without the skip flag."
        )
    print(f"Loading cached data from: {file_path}")
    return pd.read_csv(file_path)
