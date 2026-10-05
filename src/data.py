from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

RAW_PATH = Path("data/raw/credit_default.csv")
PROC_DIR = Path("data/processed")
RANDOM_STATE = 42


def download_raw():
    """Download the dataset from UCI and save it as one CSV file."""
    ds = fetch_ucirepo(id = 350)  # 350 = Default of Credit Card Clients
    df = pd.concat([ds.data.features, ds.data.targets], axis = 1)
    RAW_PATH.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(RAW_PATH, index = False)
    print("Saved raw data:", df.shape)
    print("Columns:", list(df.columns))


# X1..X23 order from the UCI documentation
UCI_NAMES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def tidy_names(df):
    """Give the columns readable names."""
    df = df.copy()
    df = df.rename(columns=dict(zip([f"X{i}" for i in range(1, 24)], UCI_NAMES)))
    df = df.rename(columns={"Y": "default"})
    return df


def split_and_save():
    df = tidy_names(pd.read_csv(RAW_PATH))
    train, test = train_test_split(
        df, test_size=0.2, stratify=df["default"], random_state=RANDOM_STATE
    )
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROC_DIR / "train.csv", index=False)
    test.to_csv(PROC_DIR / "test.csv", index=False)
    print("train:", train.shape, "default rate:", round(train["default"].mean(), 4))
    print("test: ", test.shape, "default rate:", round(test["default"].mean(), 4))


if __name__ == "__main__":
    if not RAW_PATH.exists():
        download_raw()
    split_and_save()