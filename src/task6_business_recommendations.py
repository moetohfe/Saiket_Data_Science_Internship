"""
Task 6 - Business Recommendations

Turns the findings from Tasks 2-5 into retention actions and estimates
their impact on churn and revenue.

How the estimates work
----------------------
For each action we take the churn rate of the customers we target and
the churn rate of comparable customers who already behave the way we
want (e.g. month-to-month vs one-year contracts). Customers who switch
will not instantly behave exactly like the other group, so only HALF of
that observed gap is counted (REALISATION = 0.5). The cost of each
incentive is subtracted to give a net figure. All assumptions are listed
in the generated report so they can be challenged and adjusted.
"""

import matplotlib.pyplot as plt
import pandas as pd

from common import (REPORTS_DIR, TARGET, churn_rate, ensure_dirs, header, load_clean,
                    save_fig, set_style)

REALISATION = 0.5  # share of the observed churn gap we expect to actually capture
MONTHS = 12        # impact horizon


def scenario(name, target, rate_now, rate_goal, uptake, monthly_cost_per_customer, action, why,
             cost_months=MONTHS):
    """Estimate customers and revenue saved over 12 months for one action."""
    n_target = len(target)
    converted = n_target * uptake
    reduction = (rate_now - rate_goal) / 100 * REALISATION
    saved = converted * reduction
    arpu = target["MonthlyCharges"].mean()
    revenue = saved * arpu * MONTHS
    cost = converted * monthly_cost_per_customer * cost_months
    return {"recommendation": name, "action": action, "evidence": why,
            "customers_targeted": n_target, "assumed_uptake": uptake,
            "churn_now_%": round(rate_now, 1), "churn_comparison_%": round(rate_goal, 1),
            "customers_retained": round(saved), "revenue_retained_$": round(revenue),
            "incentive_cost_$": round(cost), "net_impact_$": round(revenue - cost)}


def build_scenarios(df):
    mtm = df[df["Contract"] == "Month-to-month"]
    one_year = df[df["Contract"] == "One year"]
    echeck = df[df["PaymentMethod"] == "Electronic check"]
    auto = df[df["PaymentMethod"].str.contains("automatic")]
    fiber = df[df["InternetService"] == "Fiber optic"]
    fiber_no_support = fiber[(fiber["OnlineSecurity"] == "No") & (fiber["TechSupport"] == "No")]
    fiber_support = fiber[(fiber["OnlineSecurity"] == "Yes") | (fiber["TechSupport"] == "Yes")]
    new = df[df["tenure"] <= 12]
    developing = df[(df["tenure"] > 12) & (df["tenure"] <= 24)]

    return [
        scenario("1. Contract upgrade offer", mtm, churn_rate(mtm[TARGET]), churn_rate(one_year[TARGET]),
                 uptake=0.20, monthly_cost_per_customer=5.0,
                 action="Offer month-to-month customers a $5/month discount to move to a 1-year contract.",
                 why=f"Month-to-month churn is {churn_rate(mtm[TARGET]):.1f}% vs "
                     f"{churn_rate(one_year[TARGET]):.1f}% on 1-year contracts; contract type is a top-2 model driver."),
        scenario("2. Auto-pay migration", echeck, churn_rate(echeck[TARGET]), churn_rate(auto[TARGET]),
                 uptake=0.30, monthly_cost_per_customer=2.0,
                 action="Give electronic-check payers a $2/month credit for switching to automatic card or bank payment.",
                 why=f"Electronic-check churn is {churn_rate(echeck[TARGET]):.1f}% vs "
                     f"{churn_rate(auto[TARGET]):.1f}% for automatic payments."),
        scenario("3. Fiber 'Security & Support' bundle", fiber_no_support,
                 churn_rate(fiber_no_support[TARGET]), churn_rate(fiber_support[TARGET]),
                 uptake=0.25, monthly_cost_per_customer=5.0, cost_months=6,
                 action="Include Online Security + Tech Support free for 6 months for fiber customers who have neither.",
                 why=f"Fiber customers without security/support churn at {churn_rate(fiber_no_support[TARGET]):.1f}% "
                     f"vs {churn_rate(fiber_support[TARGET]):.1f}% with at least one."),
        scenario("4. First-year onboarding program", new, churn_rate(new[TARGET]), churn_rate(developing[TARGET]),
                 uptake=1.00, monthly_cost_per_customer=1.0,
                 action="Welcome calls, a 30/60/90-day check-in and early issue resolution for every new customer.",
                 why=f"Customers in their first year churn at {churn_rate(new[TARGET]):.1f}% vs "
                     f"{churn_rate(developing[TARGET]):.1f}% in year two; tenure is the #1 model driver."),
    ]


def impact_chart(table):
    fig, ax = plt.subplots(figsize=(11, 5))
    names = table["recommendation"]
    ax.barh(names, table["revenue_retained_$"] / 1000, color="#7FB77E", label="Revenue retained")
    ax.barh(names, -table["incentive_cost_$"] / 1000, color="#DD5555", label="Incentive cost")
    for i, net in enumerate(table["net_impact_$"]):
        ax.annotate(f"net ${net / 1000:,.0f}k", (table["revenue_retained_$"].iloc[i] / 1000, i),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=10, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("$ thousands over 12 months")
    ax.set_title("Estimated 12-Month Impact of Each Recommendation")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.2)
    save_fig(fig, "task6_01_recommendation_impact.png")


def write_report(df, table):
    base_rate = churn_rate(df[TARGET])
    churners = (df[TARGET] == "Yes").sum()
    # Actions overlap (one customer can be in several groups), so the combined
    # effect is discounted by 30% rather than simply added up.
    combined_saved = table["customers_retained"].sum() * 0.7
    new_rate = (churners - combined_saved) / len(df) * 100
    combined_net = table["net_impact_$"].sum() * 0.7

    # High-value at-risk group, defined exactly as in Task 3
    premium = pd.qcut(df["MonthlyCharges"], q=4, labels=False) == 3
    hv_targets = df[premium & (df["Contract"] == "Month-to-month") & (df[TARGET] == "No")]

    # Best model's test performance from Task 4 (first row = best by cross-validation)
    metrics_file = REPORTS_DIR / "task4_model_comparison.csv"
    if metrics_file.exists():
        best = pd.read_csv(metrics_file).iloc[0]
        model_line = (f"The {best['model']} model catches ~{best['recall']:.0%} of churners (recall) "
                      f"with ROC-AUC {best['roc_auc']:.2f}.")
    else:
        model_line = "Run Task 4 first to fill in model performance."

    lines = [
        "# Business Recommendations: Reducing Customer Churn",
        "",
        f"**Current situation:** {base_rate:.1f}% of customers churned ({churners:,} of {len(df):,}), "
        f"taking ${df.loc[df[TARGET] == 'Yes', 'MonthlyCharges'].sum():,.0f} of monthly revenue with them.",
        "",
        "## Recommended actions",
        "",
    ]
    for _, r in table.iterrows():
        lines += [
            f"### {r['recommendation']}",
            f"- **Action:** {r['action']}",
            f"- **Evidence:** {r['evidence']}",
            f"- **Who:** {r['customers_targeted']:,} customers; assumed uptake {r['assumed_uptake']:.0%}",
            f"- **Estimated 12-month impact:** ~{r['customers_retained']:,} customers retained, "
            f"${r['revenue_retained_$']:,} revenue kept, ${r['incentive_cost_$']:,} incentive cost, "
            f"**net ${r['net_impact_$']:,}**",
            "",
        ]
    lines += [
        "### 5. Model-driven early-warning list",
        "- **Action:** Score every customer monthly with the churn model (Task 4) and send the highest-risk "
        "high-value customers to a dedicated retention team.",
        f"- **Evidence:** {model_line} "
        f"Task 3 found {len(hv_targets):,} active Premium-tier customers on month-to-month contracts "
        f"worth ~${hv_targets['MonthlyCharges'].sum() / 1000:,.0f}k/month.",
        "- **Impact:** Not added to the totals to avoid double counting with actions 1-4; it is the "
        "delivery channel that makes them targeted and cheaper.",
        "",
        "## Combined impact (estimate)",
        "",
        f"Because the groups overlap, the combined effect is discounted by 30%: about "
        f"**{combined_saved:,.0f} customers retained per year**, lowering churn from **{base_rate:.1f}% to "
        f"~{new_rate:.1f}%**, for a **net gain of roughly ${combined_net / 1000:,.0f}k over 12 months**.",
        "",
        "## Assumptions",
        "",
        f"- Only {REALISATION:.0%} of each observed churn gap is assumed to be achievable "
        "(customers who switch will not fully behave like the comparison group).",
        "- Uptake rates (20-100%) and incentive costs are planning assumptions, not measured values.",
        "- Revenue = customers retained x their average monthly charge x 12 months.",
        "- Recommended next step: A/B test each offer on a small group first to measure the real uplift.",
    ]
    (REPORTS_DIR / "task6_business_recommendations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base_rate, new_rate, combined_saved, combined_net


def main():
    ensure_dirs()
    set_style()
    header("TASK 6: BUSINESS RECOMMENDATIONS")
    df = load_clean()
    table = pd.DataFrame(build_scenarios(df))
    table.to_csv(REPORTS_DIR / "task6_recommendation_impact.csv", index=False)

    show = table[["recommendation", "customers_targeted", "churn_now_%", "churn_comparison_%",
                  "customers_retained", "revenue_retained_$", "incentive_cost_$", "net_impact_$"]]
    print(show.to_string(index=False))
    impact_chart(table)

    base_rate, new_rate, saved, net = write_report(df, table)
    print(f"\nCombined (overlap-adjusted): ~{saved:,.0f} customers retained/year, "
          f"churn {base_rate:.1f}% -> ~{new_rate:.1f}%, net ~${net:,.0f} over 12 months")
    print("Saved outputs/reports/task6_business_recommendations.md and task6_recommendation_impact.csv")


if __name__ == "__main__":
    main()
