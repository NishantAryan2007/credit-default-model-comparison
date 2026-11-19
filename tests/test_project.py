import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import DROP_COLS, split_xy, clean
from src.models import CAT_COLS, MONEY_COLS

MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
EXPECTED_COLS = ["LIMIT_BAL", "EDUCATION", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] + \
    [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def train():
    return pd.read_csv("data/processed/train.csv")


def test_sensitive_columns_are_dropped(train):
    X, _ = split_xy(train)
    for col in DROP_COLS:
        assert col not in X.columns


def test_sensitive_columns_still_in_saved_data(train):
    # kept on purpose so a fairness check stays possible
    for col in DROP_COLS:
        assert col in train.columns


def test_model_inputs_are_the_expected_20_columns(train):
    X, _ = split_xy(train)
    assert list(X.columns) == EXPECTED_COLS


def test_education_codes_are_merged():
    df = pd.DataFrame({"EDUCATION": [0, 1, 2, 3, 4, 5, 6]})
    assert list(clean(df)["EDUCATION"]) == [4, 1, 2, 3, 4, 4, 4]


def test_clean_does_not_change_original(train):
    before = train.copy()
    clean(train)
    pd.testing.assert_frame_equal(train, before)


def test_models_list_columns_are_inputs():
    assert set(CAT_COLS + MONEY_COLS) == set(EXPECTED_COLS)


@pytest.mark.parametrize("name", MODELS)
def test_saved_model_returns_valid_probabilities(name, train):
    model = joblib.load(f"artifacts/models/{name}.joblib")
    X, _ = split_xy(train.head(200))
    p = model.predict_proba(X)[:, 1]
    assert len(p) == 200
    assert np.all((p >= 0) & (p <= 1))


def test_single_customer_prediction_works():
    # the app sends one row at a time
    model = joblib.load("artifacts/models/xgboost.joblib")
    row = {c: 0 for c in EXPECTED_COLS}
    row.update({"LIMIT_BAL": 100000, "EDUCATION": 2})
    p = model.predict_proba(pd.DataFrame([row]))[0, 1]
    assert 0 <= p <= 1