from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "classification" / "classification_data.csv"

TARGET = "IsCanceled"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_CATEGORY_FREQ = 50  # rare categories get grouped

NUMERICAL_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "arrival_month",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "arrival_dow",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "has_agent",
    "has_company",
]

CATEGORICAL_FEATURES = [
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "DepositType",
    "CustomerType",
]

# # these give away the answer
# LEAKY = ["ReservationStatus", "ReservationStatusDate", "AssignedRoomType", "room_changed"]
# # not needed
# REDUNDANT = ["arrival_date", "ArrivalDateMonth", "ArrivalDateYear"]

# CATEGORICAL = [
#     "Meal", "Country", "MarketSegment", "DistributionChannel", "ReservedRoomType",
#     "DepositType", "Agent", "Company", "CustomerType",
# ]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)

    return X, y


def  split_data(X, y):
    # 80% train, 20% test
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

# --- PREPROCESSORS ---

# for logistic regression, svm, knn, pca, lda
def linear_preprocessor():

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=MIN_CATEGORY_FREQ,
            ),
        ),
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


# for tree models
def tree_preprocessor():

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=MIN_CATEGORY_FREQ,
            ),
        ),
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

# # only for HistGradientBoosting
# def tree_categorical_mask(X):
#     return [False] * len(numeric_columns(X)) + [True] * len(CATEGORICAL)


# --- DATA PREPARATION ---

def prepare_classification_data():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Classification data preparation completed.")
    print("-------------------------------------------")
    print("Total samples:", len(X))
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Numerical features:", len(NUMERICAL_FEATURES))
    print("Categorical features:", len(CATEGORICAL_FEATURES))

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_classification_data()
