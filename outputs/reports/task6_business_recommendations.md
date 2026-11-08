# Business Recommendations: Reducing Customer Churn

**Current situation:** 26.5% of customers churned (1,869 of 7,043), taking $139,131 of monthly revenue with them.

## Recommended actions

### 1. Contract upgrade offer
- **Action:** Offer month-to-month customers a $5/month discount to move to a 1-year contract.
- **Evidence:** Month-to-month churn is 42.7% vs 11.3% on 1-year contracts; contract type is a top-2 model driver.
- **Who:** 3,875 customers; assumed uptake 20%
- **Estimated 12-month impact:** ~122 customers retained, $97,072 revenue kept, $46,500 incentive cost, **net $50,572**

### 2. Auto-pay migration
- **Action:** Give electronic-check payers a $2/month credit for switching to automatic card or bank payment.
- **Evidence:** Electronic-check churn is 45.3% vs 16.0% for automatic payments.
- **Who:** 2,365 customers; assumed uptake 30%
- **Estimated 12-month impact:** ~104 customers retained, $95,126 revenue kept, $17,028 incentive cost, **net $78,098**

### 3. Fiber 'Security & Support' bundle
- **Action:** Include Online Security + Tech Support free for 6 months for fiber customers who have neither.
- **Evidence:** Fiber customers without security/support churn at 55.0% vs 24.5% with at least one.
- **Who:** 1,765 customers; assumed uptake 25%
- **Estimated 12-month impact:** ~67 customers retained, $69,572 revenue kept, $13,238 incentive cost, **net $56,334**

### 4. First-year onboarding program
- **Action:** Welcome calls, a 30/60/90-day check-in and early issue resolution for every new customer.
- **Evidence:** Customers in their first year churn at 47.4% vs 28.7% in year two; tenure is the #1 model driver.
- **Who:** 2,186 customers; assumed uptake 100%
- **Estimated 12-month impact:** ~205 customers retained, $137,791 revenue kept, $26,232 incentive cost, **net $111,559**

### 5. Model-driven early-warning list
- **Action:** Score every customer monthly with the churn model (Task 4) and send the highest-risk high-value customers to a dedicated retention team.
- **Evidence:** The Random Forest model catches ~78% of churners (recall) with ROC-AUC 0.84. Task 3 found 416 active Premium-tier customers on month-to-month contracts worth ~$41k/month.
- **Impact:** Not added to the totals to avoid double counting with actions 1-4; it is the delivery channel that makes them targeted and cheaper.

## Combined impact (estimate)

Because the groups overlap, the combined effect is discounted by 30%: about **349 customers retained per year**, lowering churn from **26.5% to ~21.6%**, for a **net gain of roughly $208k over 12 months**.

## Assumptions

- Only 50% of each observed churn gap is assumed to be achievable (customers who switch will not fully behave like the comparison group).
- Uptake rates (20-100%) and incentive costs are planning assumptions, not measured values.
- Revenue = customers retained x their average monthly charge x 12 months.
- Recommended next step: A/B test each offer on a small group first to measure the real uplift.
