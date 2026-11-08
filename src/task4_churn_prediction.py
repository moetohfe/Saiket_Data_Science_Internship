"""
Task 4 - Churn Prediction Model

* Train several classification algorithms
* Feature selection and hyperparameter tuning with cross-validation
* Compare models on accuracy, precision, recall and F1-score
* Save the tuned models for Task 5
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import (MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET,
                    ensure_dirs, header, mutual_info, save_fig, set_style)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def load_split():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return (train.drop(columns=[TARGET]), train[TARGET],
            test.drop(columns=[TARGET]), test[TARGET])


def candidate_models():
    """
    Each model is a pipeline: scale -> select the k most informative
    features -> classifier. GridSearchCV tunes k (feature selection) and
    the classifier's hyperparameters together. class_weight='balanced'
    compensates for churners being only ~27% of customers.
    """
    def pipe(clf):
        return Pipeline([("scale", StandardScaler()),
                         ("select", SelectKBest(score_func=mutual_info)),
                         ("clf", clf)])

    return {
        "Logistic Regression": (
            pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
            {"select__k": [10, 20, "all"], "clf__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (
            pipe(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
            {"select__k": [10, 20, "all"], "clf__max_depth": [4, 6, 8],
             "clf__min_samples_leaf": [10, 30, 60]}),
        "Random Forest": (
            pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=RANDOM_STATE, n_jobs=-1)),
            {"select__k": [20, "all"], "clf__max_depth": [6, 10],
             "clf__min_samples_leaf": [5, 20]}),
        "Gradient Boosting": (
            pipe(GradientBoostingClassifier(random_state=RANDOM_STATE)),
            {"select__k": [20, "all"], "clf__n_estimators": [100, 200],
             "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}),
    }


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}


def comparison_chart(results):
    melted = results.reset_index().melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"],
                                        var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="model", palette="Set2", ax=ax)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison on the Test Set")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    save_fig(fig, "task4_01_model_comparison.png")


def main():
    ensure_dirs()
    set_style()
    header("TASK 4: CHURN PREDICTION MODEL")
    X_train, y_train, X_test, y_test = load_split()
    print(f"Training on {len(X_train):,} customers, testing on {len(X_test):,}; "
          f"{X_train.shape[1]} candidate features")
    print("Tuning with 5-fold stratified cross-validation, optimising F1-score...\n")

    fitted, rows = {}, []
    for name, (pipeline, grid) in candidate_models().items():
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=CV, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted[name] = best
        scores = evaluate(best, X_test, y_test)
        k = best.named_steps["select"].k
        n_feat = X_train.shape[1] if k == "all" else k
        params = {key.replace("clf__", ""): v for key, v in search.best_params_.items() if key.startswith("clf__")}
        rows.append({"model": name, "cv_f1": search.best_score_, **scores,
                     "features_used": n_feat, "best_params": params})
        print(f"{name:<20} CV F1 {search.best_score_:.3f} | test F1 {scores['f1']:.3f} | "
              f"features {n_feat} | {params}")

    results = pd.DataFrame(rows).set_index("model").sort_values("cv_f1", ascending=False)
    best_name = results.index[0]  # chosen by cross-validation, not by peeking at the test set
    print("\nTest-set results:")
    print(results[["cv_f1", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(3).to_string())
    print(f"\nBest model (highest cross-validated F1): {best_name}")

    # Baseline for context: predicting "no churn" for everyone
    print(f"Baseline 'nobody churns' accuracy: {1 - y_test.mean():.3f} with recall 0.000")

    results.round(4).to_csv(REPORTS_DIR / "task4_model_comparison.csv")
    comparison_chart(results)

    selected = fitted[best_name].named_steps["select"].get_support()
    pd.Series(X_train.columns[selected]).to_csv(REPORTS_DIR / "task4_selected_features.csv",
                                                index=False, header=["feature"])
    joblib.dump({"models": fitted, "best": best_name, "features": list(X_train.columns)},
                MODELS_DIR / "churn_models.joblib")
    print("Saved outputs/reports/task4_model_comparison.csv and outputs/models/churn_models.joblib")


if __name__ == "__main__":
    main()
