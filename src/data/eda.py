
from pathlib import Path

from networkx import display
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "H2.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

# === DATASET OVERVIEW ===
print("=== DATASET OVERVIEW ===")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
df.info()
# childern and country columns contain missing values, what do they mean? How should we handle them?
# children column has 0.5% missing values, can we fill them with 0?
# country column has 0.1% missing values,can we fill them with "Unknown"?
# children column is of type float, we will convert it to int after filling missing values
# ReservationStatusDate column is of type object, we will convert it to datetime
# Agent column is of type object, we will convert it to category
# Company column is of type object, we will convert it to category
# Company have NULL values
# Check duplicates ignoring specific tracking columns if any exist
# Should we drop the duplicates or keep them? If we keep them, how do we handle them in the model?
# NA children fill them with 0
# Replace 'Undefined' in Meal with 'SC' 

# Type encoding? do we prioritize encoding the target or the feature
# countries?

# === MISSING VALUES ===
print("\n=== MISSING VALUES ===")

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": df.isnull().mean() * 100,
}).sort_values("missing_count", ascending=False)

missing.to_csv(OUTPUT_DIR / "missing_values.csv")
print(missing[missing["missing_count"] > 0].to_string())

# === DUPLICATE ROWS ===
print("\n=== DUPLICATE ROWS ===")
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count:,}")  

# === NUMERICAL SUMMARY ===
print("\n=== NUMERICAL SUMMARY ===")
print(df.describe().T)

df.describe().T.to_csv(OUTPUT_DIR / "numerical_summary.csv")

# === REGRESSION TARGET: ADR ===
print("\n=== ADR SUMMARY ===")
print(df["ADR"].describe())

# === CLASSIFICATION TARGET: IsCanceled ===
print("\n=== CANCELLATION DISTRIBUTION ===")
print(df["IsCanceled"].value_counts().sort_index())

print("\nCancellation percentages:")
print(
    (df["IsCanceled"].value_counts(normalize=True).sort_index() * 100).round(2)
)

# === SUSPICIOUS VALUES ===
print("\n=== SUSPICIOUS VALUES ===")

if "ADR" in df.columns:
    print("Negative ADR values:", (df["ADR"] < 0).sum())

if "LeadTime" in df.columns:
    print("Negative LeadTime values:", (df["LeadTime"] < 0).sum())

if "Adults" in df.columns:
    print("Bookings with zero adults:", (df["Adults"] == 0).sum())

if "Children" in df.columns:
    print("Negative Children values:", (df["Children"] < 0).sum())

# Check children and babies for the rows where Adults == 0
zero_adults = df[df["Adults"] == 0]
print(zero_adults[["Adults", "Children", "Babies", "IsCanceled"]].head(10))
print(f"Total rows with 0 adults: {len(zero_adults)}")

# Check Zero-Adults with Children/Babies Scenario
print("\n=== Zero-Adults with Children/Babies Scenario ===")
# Filter rows where Adults == 0, but have children or babies
zero_adults_with_kids = df[
    (df["Adults"] == 0) & ((df["Children"] > 0) | (df["Babies"] > 0))
]

print(
    f"Total rows with Adults == 0 but Children/Babies > 0:"
    f" {len(zero_adults_with_kids)}"
)

if not zero_adults_with_kids.empty:
  print("\n--- Customer Type Breakdown ---")
  print(zero_adults_with_kids["CustomerType"].value_counts(dropna=False))

  print("\n--- Reserved Room Type Breakdown ---")
  print(zero_adults_with_kids["ReservedRoomType"].value_counts(dropna=False))

  print("\n--- Sample Rows Preview ---")
  print(
      zero_adults_with_kids[
          [
              "Adults",
              "Children",
              "Babies",
              "CustomerType",
              "ReservedRoomType",
              "ADR",
              "IsCanceled",
          ]
      ].head(10)
  )

# Top 10 Countries Bar Plot
top_countries = df["Country"].value_counts().head(10)
plt.figure(figsize=(10, 6))

bars = plt.barh(
    top_countries.index[::-1],
    top_countries.to_numpy()[::-1],
    color="skyblue",
    edgecolor="black",
    alpha=0.8,
)
plt.title(
    "Top 10 Countries of Origin for Hotel Bookings",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Number of Bookings", fontsize=12)
plt.ylabel("Country Code", fontsize=12)

for bar in bars:
  width = bar.get_width()
  plt.text(
      width + 200,
      bar.get_y() + bar.get_height() / 2,
      f"{width:,}",
      va="center",
      ha="left",
      fontsize=10,
      color="black",
  )

plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_countries_plot.png", dpi=300)

# === ADR Distribution Plot ===
adr_filtered = df[(df["ADR"] > 0) & (df["ADR"] < 1000)]["ADR"] # Filter out the extreme outlier (5400) for a clean visualization

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Histogram
ax1.hist(adr_filtered, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
ax1.set_title("ADR Distribution (Excluding Extreme Outliers > 1000)")
ax1.set_xlabel("Average Daily Rate (ADR)")
ax1.set_ylabel("Frequency")
ax1.grid(True, linestyle="--", alpha=0.5)

# 2. Boxplot
ax2.boxplot(adr_filtered, vert=False, patch_artist=True, 
            boxprops=dict(facecolor="skyblue", color="black"),
            medianprops=dict(color="red", linewidth=2))
ax2.set_title("ADR Boxplot")
ax2.set_xlabel("Average Daily Rate (ADR)")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "adr_distribution_plot.png", dpi=300)