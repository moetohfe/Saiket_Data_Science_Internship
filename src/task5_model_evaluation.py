"""
Task 5 - Model Evaluation and Interpretation

* Evaluate the best model on the held-out test set
* ROC curves and AUC for every model
* Interpret what drives churn: feature importances and
  logistic-regression coefficients (which show direction)
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, auc, classification_report,
                             confusion_matrix, roc_curve)

from common import (MODELS_DIR, RANDOM_STATE, REPORTS_DIR, ensure_dirs, header,
                    save_fig, set_style)
from task4_churn_prediction import load_split


def roc_curves(models, X_test, y_test, best_name):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    aucs = {}
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        aucs[name] = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=3 if name == best_name else 1.5,
                label=f"{name} (AUC = {aucs[name]:.3f})" + ("  <- best" if name == best_name else ""))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC Curves on the Test Set")
    ax.legend(loc="lower right", fontsize=9)
    save_fig(fig, "task5_01_roc_curves.png")
    return aucs


def confusion(model, X_test, y_test, name):
    pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Stayed", "Churned"]).plot(
        cmap="Blues", values_format="d", colorbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix: {name}")
    ax.grid(False)
    save_fig(fig, "task5_02_confusion_matrix.png")

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\n{name} on {len(y_test):,} test customers:")
    print(f"  Churners correctly flagged: {tp} of {tp + fn} ({tp / (tp + fn):.1%})")
    print(f"  Churners missed:            {fn}")
    print(f"  Loyal customers flagged:    {fp} (would receive an unnecessary offer)")
    print("\n" + classification_report(y_test, pred, target_names=["Stayed", "Churned"], digits=3))


def feature_importance(model, X_test, y_test, name):
    """Permutation importance: how much ROC-AUC drops when a feature is shuffled."""
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean,
                         "std": result.importances_std})
           .sort_values("importance", ascending=False))
    imp.round(4).to_csv(REPORTS_DIR / "task5_feature_importance.csv", index=False)

    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top["feature"][::-1], top["importance"][::-1], xerr=top["std"][::-1], color="#4C72B0")
    ax.set_title(f"Top 15 Churn Drivers ({name}, permutation importance)")
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    save_fig(fig, "task5_03_feature_importance.png")
    print("Top 10 features by permutation importance:")
    print(top.head(10)[["feature", "importance"]].round(4).to_string(index=False))


def coefficients(model, feature_names):
    """Logistic regression shows WHICH WAY each feature pushes churn."""
    select = model.named_steps["select"].get_support()
    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=np.array(feature_names)[select])
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = ["#DD5555" if c > 0 else "#4C72B0" for c in top[::-1]]
    ax.barh(top.index[::-1], top[::-1], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Logistic Regression Coefficients (standardised)")
    ax.set_xlabel("<- lowers churn risk          raises churn risk ->")
    save_fig(fig, "task5_04_logistic_coefficients.png")

    coefs.sort_values().round(4).to_csv(REPORTS_DIR / "task5_logistic_coefficients.csv", header=["coefficient"])
    print("\nStrongest churn-RAISING factors (logistic regression):")
    print(coefs.sort_values(ascending=False).head(5).round(3).to_string())
    print("\nStrongest churn-LOWERING factors:")
    print(coefs.sort_values().head(5).round(3).to_string())


def probability_distribution(model, X_test, y_test, name):
    proba = model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(x=proba, hue=y_test.map({0: "Stayed", 1: "Churned"}), bins=30, stat="density",
                 common_norm=False, palette={"Stayed": "#4C72B0", "Churned": "#DD5555"},
                 element="step", ax=ax)
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_title(f"Predicted Churn Probability ({name})")
    ax.set_xlabel("Predicted probability of churn")
    save_fig(fig, "task5_05_probability_distribution.png")


def main():
    ensure_dirs()
    set_style()
    header("TASK 5: MODEL EVALUATION AND INTERPRETATION")
    bundle = joblib.load(MODELS_DIR / "churn_models.joblib")
    models, best_name = bundle["models"], bundle["best"]
    _, _, X_test, y_test = load_split()
    best = models[best_name]

    aucs = roc_curves(models, X_test, y_test, best_name)
    print("ROC-AUC on the test set:")
    for name, value in sorted(aucs.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {value:.3f}")

    confusion(best, X_test, y_test, best_name)
    probability_distribution(best, X_test, y_test, best_name)
    feature_importance(best, X_test, y_test, best_name)
    coefficients(models["Logistic Regression"], bundle["features"])
    print("\nSaved outputs/reports/task5_feature_importance.csv and task5_logistic_coefficients.csv")


if __name__ == "__main__":
    main()
