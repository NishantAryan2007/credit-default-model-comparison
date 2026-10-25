import pandas as pd

TARGET = "default"
# Kept in the saved CSVs for auditing, never given to the models
DROP_COLS = ["SEX", "MARRIAGE", "AGE"]


def clean(df):
    """Same cleaning for train, test, and (later) the app's input."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    return df.drop(columns=DROP_COLS, errors="ignore")


def split_xy(df):
    """Return model inputs X and the target y."""
    df = clean(df)
    return df.drop(columns=TARGET), df[TARGET]