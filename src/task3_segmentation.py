"""
Task 3 - Customer Segmentation

* Segment customers by tenure, monthly charges and contract type
* Compare churn rates across segments
* Identify high-value customers who are at risk of churning
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (CHURN_COLOR, REPORTS_DIR, TARGET, churn_rate, ensure_dirs,
                    header, load_clean, save_fig, set_style)

TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["New (0-12m)", "Developing (13-24m)", "Established (25-48m)", "Loyal (49-72m)"]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TenureSegment"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS,
                                 include_lowest=True)
    # Spending tiers based on quartiles of monthly charges
    df["SpendSegment"] = pd.qcut(df["MonthlyCharges"], q=4,
                                 labels=["Low", "Medium", "High", "Premium"])
    df["ChurnFlag"] = (df[TARGET] == "Yes").astype(int)
    return df


def segment_table(df, cols):
    return (df.groupby(cols, observed=True)
              .agg(customers=("ChurnFlag", "size"),
                   churn_rate=("ChurnFlag", lambda s: s.mean() * 100),
                   avg_monthly=("MonthlyCharges", "mean"),
                   monthly_revenue=("MonthlyCharges", "sum"))
              .round(1))


def single_segment_charts(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, order, title in [
        (axes[0], "TenureSegment", TENURE_LABELS, "Churn Rate by Tenure Segment"),
        (axes[1], "SpendSegment", ["Low", "Medium", "High", "Premium"], "Churn Rate by Monthly Spend"),
        (axes[2], "Contract", CONTRACT_ORDER, "Churn Rate by Contract"),
    ]:
        t = segment_table(df, col).reindex(order).reset_index()
        sns.barplot(data=t, x=col, y="churn_rate", color=CHURN_COLOR, ax=ax)
        for p, n in zip(ax.patches, t["customers"]):
            ax.annotate(f"{p.get_height():.1f}%\n(n={n:,})", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
        ax.set_ylim(0, t["churn_rate"].max() * 1.25)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Churn rate (%)")
        ax.set_xticks(range(len(order)), [str(o).replace(" (", "\n(") for o in order])
    fig.tight_layout()
    save_fig(fig, "task3_01_segments.png")

    edges = pd.qcut(df["MonthlyCharges"], q=4, retbins=True)[1]
    print("Monthly spend tiers ($): " + ", ".join(
        f"{lab} {edges[i]:.0f}-{edges[i + 1]:.0f}" for i, lab in enumerate(["Low", "Medium", "High", "Premium"])))
    for col in ["TenureSegment", "SpendSegment", "Contract"]:
        print(f"\nBy {col}:\n{segment_table(df, col).to_string()}")


def heatmaps(df):
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    for ax, col, order in [(axes[0], "TenureSegment", TENURE_LABELS),
                           (axes[1], "SpendSegment", ["Low", "Medium", "High", "Premium"])]:
        pivot = (df.pivot_table(index="Contract", columns=col, values="ChurnFlag",
                                aggfunc="mean", observed=True) * 100).reindex(CONTRACT_ORDER)[order]
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=70,
                    cbar_kws={"label": "Churn rate (%)"}, ax=ax)
        ax.set_title(f"Churn Rate (%): Contract x {col.replace('Segment', ' Segment')}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([str(o).replace(" (", "\n(") for o in order], rotation=0)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_fig(fig, "task3_02_segment_heatmaps.png")


def three_way_segments(df):
    seg = segment_table(df, ["Contract", "TenureSegment", "SpendSegment"]).reset_index()
    seg = seg[seg["customers"] >= 30]  # ignore tiny groups
    seg.sort_values("churn_rate", ascending=False).to_csv(
        REPORTS_DIR / "task3_segment_churn_rates.csv", index=False)
    print("\nTop 10 riskiest segments (min 30 customers):")
    print(seg.sort_values("churn_rate", ascending=False).head(10).to_string(index=False))
    print("Saved outputs/reports/task3_segment_churn_rates.csv")


def high_value_at_risk(df):
    """
    High-value = top 25% of monthly charges (Premium tier).
    At risk    = on a month-to-month contract (the strongest churn driver in Task 2).
    These customers are still with the company, so they can still be saved.
    """
    high_value = df["SpendSegment"] == "Premium"
    premium_cut = df.loc[high_value, "MonthlyCharges"].min()
    at_risk = df["Contract"] == "Month-to-month"

    hv = df[high_value]
    hv_risk = df[high_value & at_risk]
    active_targets = hv_risk[hv_risk[TARGET] == "No"]

    print(f"\nHigh-value threshold: MonthlyCharges >= ${premium_cut:.2f}")
    print(f"High-value customers: {len(hv):,}  (churn rate {churn_rate(hv[TARGET]):.1f}%)")
    print(f"High-value on month-to-month: {len(hv_risk):,}  (churn rate {churn_rate(hv_risk[TARGET]):.1f}%)")
    print(f"  -> active customers to target now: {len(active_targets):,}, "
          f"worth ${active_targets['MonthlyCharges'].sum():,.0f}/month "
          f"(${active_targets['MonthlyCharges'].sum() * 12:,.0f}/year)")

    cols = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract",
            "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]
    active_targets.sort_values("MonthlyCharges", ascending=False)[cols].to_csv(
        REPORTS_DIR / "task3_high_value_at_risk_customers.csv", index=False)
    print("Saved outputs/reports/task3_high_value_at_risk_customers.csv")

    # Revenue lost to churn, by segment
    fig, ax = plt.subplots(figsize=(10, 5))
    lost = (df[df[TARGET] == "Yes"].groupby(["SpendSegment", "Contract"], observed=True)["MonthlyCharges"]
            .sum().unstack().reindex(columns=CONTRACT_ORDER))
    lost.plot(kind="bar", stacked=True, color=["#DD5555", "#F2A65A", "#7FB77E"], ax=ax)
    ax.set_title("Monthly Revenue Lost to Churn by Spend Tier and Contract")
    ax.set_ylabel("Lost monthly revenue ($)")
    ax.set_xlabel("Monthly spend tier")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.legend(title="Contract")
    save_fig(fig, "task3_03_revenue_lost.png")

    total_lost = df.loc[df[TARGET] == "Yes", "MonthlyCharges"].sum()
    premium_share = df.loc[(df[TARGET] == "Yes") & high_value, "MonthlyCharges"].sum() / total_lost * 100
    print(f"\nMonthly revenue lost to churn: ${total_lost:,.0f}; "
          f"{premium_share:.0f}% of it came from Premium-tier customers")


def main():
    ensure_dirs()
    set_style()
    header("TASK 3: CUSTOMER SEGMENTATION")
    df = add_segments(load_clean())
    single_segment_charts(df)
    heatmaps(df)
    three_way_segments(df)
    high_value_at_risk(df)


if __name__ == "__main__":
    main()
