
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "H2_clean.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "IsCanceled"

# Initial feature selection

# What we are building:
# A classification model that predicts whether a hotel reservation will eventually be cancelled,
# using only information available at the time the reservation is initially created.

# For now im not adding: Country, Agent, or Company
# will evaluate them as additional features later

# First-pass
CLASSIFICATION_FEATURES = [

    "LeadTime",
    "ArrivalDateYear",
    "arrival_month",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "arrival_dow",

    # Stay and guest information
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",

    # Booking characteristics
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "DepositType",
    "CustomerType",

    # Guest history
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",

    # Whether an agent/company was involved
    "has_agent",
    "has_company",
]


EXCLUDED_FEATURES = {
    "BookingChanges": "May count changes made after the booking was created.",
    "AssignedRoomType": "Final assigned room may differ from the reserved room.",
    "ReservationStatus": "Directly reveals the booking outcome.",
    "ReservationStatusDate": "Records the date of the final status event.",
    "RequiredCarParkingSpaces": "May reflect information added or changed later.",
    "TotalOfSpecialRequests": "Requests may be added or changed after booking.",
    "DaysInWaitingList": "Represents a later waiting-list process (can happen after the initial booking request).",
    "ADR": "Needs a timing audit before deciding whether its recorded value is available at the chosen prediction point.",
    "Country": "The source paper warns that this may not be known until check-in.",
    "Agent": "The recorded agent ID needs a timing and usefulness review.",
    "Company": "The recorded company ID needs a timing and usefulness review.",
}


def select_classification_features(df):
    """
    Select the input features and target for the classification task.
    """

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    missing_features = [
        col for col in CLASSIFICATION_FEATURES
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "These expected classification features are missing:\n"
            + "\n".join(missing_features)
        )

    X = df[CLASSIFICATION_FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


def prepare_classification_data():
    df = pd.read_csv(INPUT_PATH)

    X, y = select_classification_features(df)

    # Combine X and y for a convenient classification-ready dataset.
    classification_df = X.copy()
    classification_df[TARGET] = y

    output_path = OUTPUT_DIR / "classification_data.csv"
    classification_df.to_csv(output_path, index=False)

    print("Classification feature selection completed.")
    print("Selected features:", len(X.columns))
    print("Dataset shape:", classification_df.shape)
    print("Target distribution:")
    print(y.value_counts())
    print(f"Saved to: {output_path}")

    return X, y


if __name__ == "__main__":
    prepare_classification_data()
