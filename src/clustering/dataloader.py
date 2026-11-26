from pathlib import Path
import pandas as pd

# Resolve project root dynamically (City-Hotel-Booking)
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "H2_clean_linearreg.csv"


def load_raw_cleaned_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Loads the cleaned dataset with duplicates intact (78,890 rows)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_raw_cleaned_data()
    print(f"Dataset Shape: {df.shape}\n")
    print(f"{'#':<3} | {'Column Name':<28} | {'Dtype':<10} | {'Nulls':<6} | {'Unique':<7}")
    print("-" * 65)
    for idx, col in enumerate(df.columns):
        dtype = str(df[col].dtype)
        nulls = df[col].isna().sum()
        nunique = df[col].nunique(dropna=True)
        print(f"{idx:<3} | {col:<28} | {dtype:<10} | {nulls:<6} | {nunique:<7}")