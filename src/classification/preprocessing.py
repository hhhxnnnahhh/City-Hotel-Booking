from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "H2_clean.csv"

TARGET = "IsCanceled"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_CATEGORY_FREQ = 50  # rare categories get grouped

# these give away the answer
LEAKY = ["ReservationStatus", "ReservationStatusDate", "AssignedRoomType", "room_changed"]
# not needed
REDUNDANT = ["arrival_date", "ArrivalDateMonth", "ArrivalDateYear"]

CATEGORICAL = [
    "Meal", "Country", "MarketSegment", "DistributionChannel", "ReservedRoomType",
    "DepositType", "Agent", "Company", "CustomerType",
]


def load_data():
    # agent and company are ids, not numbers
    df = pd.read_csv(DATA_PATH, dtype={"Agent": str, "Company": str})
    X = df.drop(columns=[TARGET] + LEAKY + REDUNDANT)
    y = df[TARGET].astype(int)
    return X, y


def split(X, y):
    # 80% train, 20% test
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def numeric_columns(X):
    return [c for c in X.columns if c not in CATEGORICAL]


# for logistic regression, svm, knn, pca, lda
def linear_preprocessor(X):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_columns(X)),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=MIN_CATEGORY_FREQ,
                              sparse_output=False), CATEGORICAL),
    ])


# for tree models
def tree_preprocessor(X):
    return ColumnTransformer([
        ("num", "passthrough", numeric_columns(X)),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               min_frequency=MIN_CATEGORY_FREQ), CATEGORICAL),
    ])


# only for HistGradientBoosting
def tree_categorical_mask(X):
    return [False] * len(numeric_columns(X)) + [True] * len(CATEGORICAL)
