import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SKEWED_NUMERIC_COLS = [
    "LeadTime",
    "ADR",
    "total_nights",
    "waiting_lead_ratio",
]

REGULAR_NUMERIC_COLS = [
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "total_guests",
    "TotalOfSpecialRequests",
    "BookingChanges",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "arrival_month",
    "arrival_dow",
]

BINARY_FLAG_COLS = [
    "has_children",
    "has_agent",
]

CATEGORICAL_COLS = [
    "MarketSegment",
    "CustomerType",
    "DepositType",
    "Meal",
    "ReservedRoomType",
]

SELECTED_FEATURES = (
    SKEWED_NUMERIC_COLS
    + REGULAR_NUMERIC_COLS
    + BINARY_FLAG_COLS
    + CATEGORICAL_COLS
)


def _clip_and_log1p(X: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(X, a_min=0, a_max=None))


def build_clustering_preprocessor(
    drop_first_ohe: bool = False,
    scale_binary: bool = False,
) -> ColumnTransformer:
    skewed_pipeline = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(_clip_and_log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_transformer = StandardScaler() if scale_binary else "passthrough"

    categorical_transformer = OneHotEncoder(
        drop="first" if drop_first_ohe else None,
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("skewed_num", skewed_pipeline, SKEWED_NUMERIC_COLS),
            ("regular_num", StandardScaler(), REGULAR_NUMERIC_COLS),
            ("binary_flags", binary_transformer, BINARY_FLAG_COLS),
            ("categorical", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def plot_preprocessing_diagnostics(
    df: pd.DataFrame,
    X_processed: np.ndarray,
    output_dir: str = "outputs/figures",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Before vs After log1p for skewed continuous features
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row_idx, col in enumerate(["LeadTime", "ADR"]):
        raw_vals = np.clip(df[col].values, a_min=0, a_max=None)
        log_vals = np.log1p(raw_vals)

        axes[row_idx, 0].hist(raw_vals, bins=50, color="#4C72B0", edgecolor="white")
        axes[row_idx, 0].set_title(f"{col} — Raw Distribution (Skewed)")
        axes[row_idx, 0].set_ylabel("Bookings")

        axes[row_idx, 1].hist(log_vals, bins=50, color="#55A868", edgecolor="white")
        axes[row_idx, 1].set_title(f"{col} — After log1p Transformation")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_log1p_before_after.png"), dpi=200)
    plt.close(fig)

    # Plot 2: Scale comparison before vs after StandardScaler
    numeric_cols = SKEWED_NUMERIC_COLS + REGULAR_NUMERIC_COLS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].boxplot(
        [df[c].values for c in numeric_cols],
        labels=numeric_cols,
        vert=False,
        showfliers=False,
    )
    axes[0].set_title("Numeric Features — Before Scaling (Raw Units)")

    axes[1].boxplot(
        [X_processed[:, i] for i in range(len(numeric_cols))],
        labels=numeric_cols,
        vert=False,
        showfliers=False,
    )
    axes[1].set_title("Numeric Features — After log1p + StandardScaler (z-score)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_scaling_before_after.png"), dpi=200)
    plt.close(fig)

    # Plot 3: PCA Cumulative Explained Variance on X_processed
    pca = PCA().fit(X_processed)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=4, color="#C44E52")
    ax.axhline(0.90, color="gray", linestyle="--", label="90% Variance Threshold")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("PCA Scree Check on Processed Clustering Matrix")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_pca_scree_check.png"), dpi=200)
    plt.close(fig)


def prepare_clustering_data(
    csv_path: str,
    drop_first_ohe: bool = False,
    scale_binary: bool = False,
    save_plots: bool = True,
    output_dir: str = "outputs/figures",
) -> tuple[pd.DataFrame, np.ndarray, list[str], ColumnTransformer]:
    df = pd.read_csv(csv_path)
    X_raw = df[SELECTED_FEATURES].copy()

    preprocessor = build_clustering_preprocessor(
        drop_first_ohe=drop_first_ohe,
        scale_binary=scale_binary,
    )
    X_processed = preprocessor.fit_transform(X_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    if save_plots:
        plot_preprocessing_diagnostics(df, X_processed, output_dir=output_dir)

    return df, X_processed, feature_names, preprocessor


#if __name__ == "__main__":
#    df, X_proc, feat_names, _ = prepare_clustering_data("H2_clean_linearreg.csv")
#   print(f"Processed shape: {X_proc.shape}")
#   print(f"Nulls: {np.isnan(X_proc).sum()} | Infs: {np.isinf(X_proc).sum()}")
#   print(f"Total encoded features: {len(feat_names)}")

from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_path = PROJECT_ROOT /"data"/"processed"/"H2_clean_linearreg.csv" 
    out_dir = PROJECT_ROOT / "outputs" / "figures"

    df, X_proc, feat_names, _ = prepare_clustering_data(
        str(data_path), output_dir=str(out_dir)
    )
    print(f"Processed shape: {X_proc.shape}")
    print(f"Nulls: {np.isnan(X_proc).sum()} | Infs: {np.isinf(X_proc).sum()}")
    print(f"Total encoded features: {len(feat_names)}")