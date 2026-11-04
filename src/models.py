import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

def build_tree(max_depth = None, min_samples_leaf = 1):
    return DecisionTreeClassifier(
        max_depth = max_depth, min_samples_leaf = min_samples_leaf, random_state = 42
    )

def build_forest(n_estimators = 300, max_depth = None, min_samples_leaf = 1):
    return RandomForestClassifier(
        n_estimators = n_estimators, max_depth = max_depth,
        min_samples_leaf = min_samples_leaf, n_jobs = -1, random_state = 42,
    )
# n_estimators is the number of trees. n_jobs = -1 uses all the CPU cores, so it runs faster.

def build_xgb(n_estimators = 200, max_depth = 3, learning_rate = 0.1):
    return XGBClassifier(
        n_estimators = n_estimators, max_depth = max_depth,
        learning_rate = learning_rate, eval_metric = "logloss",
        n_jobs = -1, random_state = 42,
    )
# eval_metric = "logloss" is just the score XGBoost tracks internally.