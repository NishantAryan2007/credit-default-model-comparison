import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CAT_COLS = ["EDUCATION", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
MONEY_COLS = (
    ["LIMIT_BAL"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_dummy():
    """Always predicts the overall default rate. This is 'no skill'."""
    return DummyClassifier(strategy = "prior")


def build_logreg(C = 1.0):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown = "infrequent_if_exist", min_frequency = 100), CAT_COLS),
        ("money", Pipeline([
            ("log", FunctionTransformer(signed_log)),
            ("scale", StandardScaler()),
        ]), MONEY_COLS),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(C = C, max_iter = 1000))])