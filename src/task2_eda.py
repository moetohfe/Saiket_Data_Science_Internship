"""
Task 2 - Exploratory Data Analysis (EDA)

* Overall churn rate
* Customer distribution by gender, partner and dependent status
* Tenure distribution and its relationship with churn
* Churn by contract type and payment method
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (CHURN_COLOR, PALETTE, REPORTS_DIR, STAYED_COLOR, TARGET,
                    churn_rate, ensure_dirs, header, load_clean, save_fig, set_style)


def rate_table(df, col):
    """Customers and churn rate for each value of a column."""
    t = df.groupby(col, observed=True)[TARGET].agg(customers="size", churn_rate=churn_rate)
    t["share_of_customers"] = t["customers"] / len(df) * 100
    return t.round(1).sort_values("churn_rate", ascending=False)


def bar_with_labels(ax, data, x, y, color, order=None):
    sns.barplot(data=data, x=x, y=y, color=color, order=order, ax=ax)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, data[y].max() * 1.15)  # headroom so labels never touch the frame


def overall_churn(df):
    rate = churn_rate(df[TARGET])
    print(f"Overall churn rate: {rate:.1f}%  ({(df[TARGET] == 'Yes').sum():,} of {len(df):,} customers)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    counts = df[TARGET].value_counts().reindex(["No", "Yes"])
    ax1.pie(counts, labels=["Stayed", "Churned"], autopct="%1.1f%%", startangle=90,
            colors=[STAYED_COLOR, CHURN_COLOR], wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax1.set_title("Overall Churn Rate")
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette=PALETTE, order=["No", "Yes"], legend=False, ax=ax2)
    ax2.set_xticks([0, 1], ["Stayed", "Churned"])
    ax2.set_xlabel("")
    ax2.set_ylabel("Customers")
    ax2.set_title("Customer Count")
    for p in ax2.patches:
        ax2.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    save_fig(fig, "task2_01_overall_churn.png")
    return rate


def demographics(df):
    cols = ["gender", "Partner", "Dependents", "SeniorCitizen"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, col in enumerate(cols):
        order = sorted(df[col].unique())  # same category order in both rows
        sns.countplot(data=df, x=col, hue=TARGET, hue_order=["No", "Yes"], palette=PALETTE,
                      order=order, ax=axes[0, i])
        axes[0, i].set_title(f"Customers by {col}")
        axes[0, i].set_ylabel("Customers")
        axes[0, i].set_ylim(0, axes[0, i].get_ylim()[1] * 1.3)
        handles, _ = axes[0, i].get_legend_handles_labels()
        axes[0, i].legend(handles, ["Stayed", "Churned"], title="Churn", ncol=2, loc="upper center")
        rates = rate_table(df, col).reset_index()
        bar_with_labels(axes[1, i], rates, col, "churn_rate", CHURN_COLOR, order=order)
        axes[1, i].set_title(f"Churn Rate by {col}")
        axes[1, i].set_ylabel("Churn rate (%)")
        axes[1, i].set_ylim(0, 50)
    fig.suptitle("Customer Demographics and Churn", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "task2_02_demographics.png")

    for col in cols:
        print(f"\nChurn by {col}:\n{rate_table(df, col).to_string()}")


def tenure_analysis(df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    sns.histplot(data=df, x="tenure", hue=TARGET, palette=PALETTE, bins=36,
                 multiple="stack", hue_order=["No", "Yes"], ax=axes[0])
    axes[0].get_legend().set_title("")
    for text, label in zip(axes[0].get_legend().get_texts(), ["Stayed", "Churned"]):
        text.set_text(label)
    axes[0].set_title("Tenure Distribution")
    axes[0].set_xlabel("Tenure (months)")

    sns.boxplot(data=df, x=TARGET, y="tenure", hue=TARGET, palette=PALETTE, order=["No", "Yes"],
                legend=False, ax=axes[1])
    axes[1].set_xticks([0, 1], ["Stayed", "Churned"])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Tenure (months)")
    axes[1].set_title("Tenure: Stayed vs Churned")

    bins = [0, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df = df.assign(tenure_band=pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True))
    rates = rate_table(df, "tenure_band").reindex(labels).reset_index()
    bar_with_labels(axes[2], rates, "tenure_band", "churn_rate", CHURN_COLOR)
    axes[2].set_title("Churn Rate by Tenure (months)")
    axes[2].set_xlabel("Tenure band")
    axes[2].set_ylabel("Churn rate (%)")
    fig.tight_layout()
    save_fig(fig, "task2_03_tenure.png")

    med = df.groupby(TARGET)["tenure"].median()
    print(f"\nMedian tenure - stayed: {med['No']:.0f} months, churned: {med['Yes']:.0f} months")
    print(f"Churn by tenure band:\n{rates.set_index('tenure_band').to_string()}")


def contract_and_payment(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    order_c = ["Month-to-month", "One year", "Two year"]
    rates_c = rate_table(df, "Contract").reindex(order_c).reset_index()
    bar_with_labels(axes[0], rates_c, "Contract", "churn_rate", CHURN_COLOR)
    axes[0].set_title("Churn Rate by Contract Type")
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_xlabel("")

    rates_p = rate_table(df, "PaymentMethod").reset_index()
    rates_p["PaymentMethod"] = rates_p["PaymentMethod"].str.replace(" (automatic)", "\n(automatic)")
    bar_with_labels(axes[1], rates_p, "PaymentMethod", "churn_rate", CHURN_COLOR)
    axes[1].set_title("Churn Rate by Payment Method")
    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_xlabel("")
    fig.tight_layout()
    save_fig(fig, "task2_04_contract_payment.png")

    print(f"\nChurn by contract:\n{rate_table(df, 'Contract').to_string()}")
    print(f"\nChurn by payment method:\n{rate_table(df, 'PaymentMethod').to_string()}")


def services_heatmap(df):
    """Bonus: churn rate for every service option, to spot risky products."""
    services = ["InternetService", "OnlineSecurity", "TechSupport", "OnlineBackup",
                "DeviceProtection", "StreamingTV", "PaperlessBilling"]
    rows = []
    for col in services:
        for val, r in rate_table(df, col)["churn_rate"].items():
            if val == "No internet service":  # same group as InternetService = No
                continue
            rows.append({"feature": f"{col} = {val}", "churn_rate": r})
    table = pd.DataFrame(rows).sort_values("churn_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(data=table, y="feature", x="churn_rate", hue="churn_rate", palette="Reds",
                legend=False, ax=ax)
    ax.axvline(churn_rate(df[TARGET]), color="black", ls="--", lw=1, label="Overall churn rate")
    ax.set_title("Churn Rate by Service Option")
    ax.set_xlabel("Churn rate (%)")
    ax.set_ylabel("")
    ax.legend()
    save_fig(fig, "task2_05_services.png")


def correlation(df):
    num = df[["tenure", "MonthlyCharges", "TotalCharges"]].assign(Churn=(df[TARGET] == "Yes").astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation of Numeric Features")
    save_fig(fig, "task2_06_correlation.png")


def main():
    ensure_dirs()
    set_style()
    header("TASK 2: EXPLORATORY DATA ANALYSIS")
    df = load_clean()
    overall_churn(df)
    demographics(df)
    tenure_analysis(df)
    contract_and_payment(df)
    services_heatmap(df)
    correlation(df)

    summary = pd.concat({c: rate_table(df, c) for c in
                         ["gender", "Partner", "Dependents", "SeniorCitizen", "Contract",
                          "PaymentMethod", "InternetService"]})
    summary.to_csv(REPORTS_DIR / "task2_churn_rates.csv")
    print("\nSaved outputs/reports/task2_churn_rates.csv")


if __name__ == "__main__":
    main()
