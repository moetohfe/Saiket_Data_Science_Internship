"""
Shared helpers for the Customer Churn Analysis project.

Every task script imports from here so that paths, data cleaning and
plot styling are defined in exactly one place.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render charts to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "Telco_Customer_Churn_Dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR = ROOT / "outputs" / "models"

RANDOM_STATE = 42
TARGET = "Churn"

# Colours used consistently across every chart
STAYED_COLOR = "#4C72B0"
CHURN_COLOR = "#DD5555"
PALETTE = {"No": STAYED_COLOR, "Yes": CHURN_COLOR}


def ensure_dirs():
    for d in (PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_raw() -> pd.DataFrame:
    """Load the raw CSV exactly as provided."""
    return pd.read_csv(RAW_DATA)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data-quality issues found during exploration (see Task 1):

    * TotalCharges is stored as text and is blank for 11 brand-new
      customers (tenure = 0). They have not been billed yet, so the
      correct value is 0, not a missing value to drop or average.
    * SeniorCitizen is 0/1 while every other yes/no column is text;
      it is mapped to "No"/"Yes" for consistent analysis.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


def mutual_info(X, y):
    """Feature-selection score used inside the model pipelines (Task 4).

    Defined here, not in the task script, so saved models can be loaded
    again by other scripts.
    """
    from sklearn.feature_selection import mutual_info_classif
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
    })


def save_fig(fig, name: str):
    ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved figure -> {path.relative_to(ROOT)}")


def churn_rate(series: pd.Series) -> float:
    """Percentage of 'Yes' values in a Churn column."""
    return (series == "Yes").mean() * 100


def header(title: str):
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")
