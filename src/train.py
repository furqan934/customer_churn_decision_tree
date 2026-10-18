"""Train and evaluate churn classifiers.

Educational project: the included CSV is synthetic. Use real, authorized data for
any business conclusions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_synthetic.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET = "churn"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV and check that the target column exists and is binary."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place a CSV in the data folder and update DATA_PATH in src/train.py."
        )
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column '{TARGET}'. Found: {list(data.columns)}")
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(int)
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The churn target must contain only 0 (retained) and 1 (churned).")
    return data


def build_models(numeric_features):
    """Build leakage-safe preprocessing and model pipelines."""
    # Median imputation is fitted on training data only because it is inside Pipeline.
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features)
        ],
        remainder="drop"
    )
    logistic_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features)
        ],
        remainder="drop"
    )

    decision_tree = Pipeline([
        ("preprocess", tree_preprocessor),
        ("model", DecisionTreeClassifier(
            criterion="gini",
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    logistic_regression = Pipeline([
        ("preprocess", logistic_preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
    return {
        "Decision Tree": decision_tree,
        "Logistic Regression": logistic_regression
    }


def evaluate_model(name, model, X_test, y_test):
    """Print and return common binary-classification metrics."""
    predictions = model.predict(X_test)
    results = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_churn": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall_churn": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1_churn": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    }
    print(f"\n{'=' * 65}\n{name}\n{'=' * 65}")
    print(f"Accuracy       : {results['accuracy']:.3f}")
    print(f"Precision (1)  : {results['precision_churn']:.3f}")
    print(f"Recall (1)     : {results['recall_churn']:.3f}")
    print(f"F1-score (1)   : {results['f1_churn']:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=[0, 1],
                                target_names=["Retained (0)", "Churned (1)"],
                                zero_division=0))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(results["confusion_matrix"]))
    return results


def save_confusion_matrix(name, model, X_test, y_test):
    """Save a confusion matrix plot for a model."""
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Retained (0)", "Churned (1)"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{name}: Confusion Matrix")
    fig.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    fig.savefig(FIGURES_DIR / f"{safe_name}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_feature_importance(model, feature_names):
    """Save feature importance for the fitted Decision Tree."""
    tree = model.named_steps["model"]
    importances = tree.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[order], importances[order])
    ax.set_xlabel("Feature importance (impurity-based)")
    ax.set_title("Decision Tree Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)

    importance_table = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    importance_table.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print("\nFeature importance (larger means more useful to this fitted tree):")
    print(importance_table.to_string(index=False))


def save_tree_plot(model, feature_names):
    """Save a readable visualization of the trained decision tree."""
    tree = model.named_steps["model"]
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=list(feature_names),
        class_names=["Retained (0)", "Churned (1)"],
        filled=True,
        rounded=True,
        proportion=False,
        impurity=True,
        max_depth=4,
        fontsize=8,
        ax=ax
    )
    ax.set_title("Decision Tree for Customer Churn (maximum depth = 4)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "decision_tree.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data(DATA_PATH)
    print("Dataset shape:", data.shape)
    print("\nFirst five rows:")
    print(data.head().to_string(index=False))
    print("\nMissing values before imputation:")
    print(data.isna().sum().to_string())
    print("\nTarget distribution (0=retained, 1=churned):")
    print(data[TARGET].value_counts().sort_index().to_string())

    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    # This starter CSV contains numerical predictors. If using categorical columns,
    # add OneHotEncoder to the preprocessing pipelines.
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_features) != X.shape[1]:
        raise ValueError("This starter script expects numeric features. Encode categorical columns first.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTraining rows: {len(X_train)} | Testing rows: {len(X_test)}")

    models = build_models(numeric_features)
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test))
        save_confusion_matrix(name, model, X_test, y_test)

    results_df = pd.DataFrame([
        {k: v for k, v in row.items() if k != "confusion_matrix"} for row in results
    ])
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    tree_model = models["Decision Tree"]
    save_feature_importance(tree_model, numeric_features)
    save_tree_plot(tree_model, numeric_features)

    print("\nModel comparison:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved figures to: {FIGURES_DIR}")
    print(f"Saved metrics and tables to: {REPORTS_DIR}")
    print("\nReminder: these results come from a synthetic educational dataset.")


if __name__ == "__main__":
    main()
