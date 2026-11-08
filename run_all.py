"""
Run the complete churn analysis pipeline (Tasks 1-6) in order.

    python run_all.py

Charts are written to outputs/figures and tables/reports to outputs/reports.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import task1_data_preparation  # noqa: E402
import task2_eda  # noqa: E402
import task3_segmentation  # noqa: E402
import task4_churn_prediction  # noqa: E402
import task5_model_evaluation  # noqa: E402
import task6_business_recommendations  # noqa: E402

STEPS = [
    ("Task 1 - Data Preparation", task1_data_preparation.main),
    ("Task 2 - Exploratory Data Analysis", task2_eda.main),
    ("Task 3 - Customer Segmentation", task3_segmentation.main),
    ("Task 4 - Churn Prediction Model", task4_churn_prediction.main),
    ("Task 5 - Model Evaluation", task5_model_evaluation.main),
    ("Task 6 - Business Recommendations", task6_business_recommendations.main),
]

if __name__ == "__main__":
    start = time.time()
    for name, step in STEPS:
        t = time.time()
        step()
        print(f"\n[done] {name} in {time.time() - t:.1f}s")
    print(f"\nAll tasks completed in {time.time() - start:.0f}s")
