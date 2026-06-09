"""
Gym Membership Churn Prediction
================================
Predicts whether a gym member will cancel their membership
using a Random Forest Classifier.

Dataset: gym_churn.csv (from Kaggle or similar source)
Target: Churn (1 = Yes, 0 = No)

Group 5 — Wei Han Lee, Kittinat Tongmai, Jihye Kim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────
# 2. Exploratory Data Analysis
# ─────────────────────────────────────────
def eda(df: pd.DataFrame) -> None:
    """Basic EDA: missing values, class distribution, correlation heatmap."""
    print("\n── Missing Values ──")
    print(df.isnull().sum())

    print("\n── Class Distribution ──")
    print(df["Churn"].value_counts())

    # Class distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["Churn"].value_counts().plot(
        kind="bar", ax=axes[0], color=["steelblue", "tomato"], edgecolor="black"
    )
    axes[0].set_title("Churn Class Distribution (Before SMOTE)")
    axes[0].set_xticklabels(["No Churn (0)", "Churn (1)"], rotation=0)
    axes[0].set_ylabel("Count")

    # Correlation heatmap
    corr = df.corr()
    sns.heatmap(
        corr,
        ax=axes[1],
        cmap="coolwarm",
        annot=False,
        linewidths=0.5,
    )
    axes[1].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("eda_overview.png", dpi=150)
    plt.show()
    print("EDA plot saved → eda_overview.png")


# ─────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Steps:
      1. Drop rows with missing values
      2. Split features / target
      3. Handle class imbalance with SMOTE
      4. Standardize features with StandardScaler
      5. Train/test split (80/20)
    """
    # Drop missing values
    df = df.dropna()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Train/test split BEFORE SMOTE (to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Train class distribution:\n{y_train.value_counts()}")

    # SMOTE on training set only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE — Train class distribution:\n{pd.Series(y_train_res).value_counts()}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    return X_train_res, X_test_scaled, y_train_res, y_test, scaler


# ─────────────────────────────────────────
# 4. Model Training & Evaluation
# ─────────────────────────────────────────
def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Train, predict, and report metrics for a given model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return {"name": name, "model": model, "accuracy": acc, "report": report, "y_pred": y_pred}


def compare_models(X_train, X_test, y_train, y_test) -> list:
    """Train and compare all four candidate models."""
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42)),
        ("SVM",                 SVC(kernel="rbf", random_state=42)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]

    results = []
    for name, model in models:
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    return results


# ─────────────────────────────────────────
# 5. Visualizations
# ─────────────────────────────────────────
def plot_model_comparison(results: list) -> None:
    """Bar chart comparing model accuracies."""
    names = [r["name"] for r in results]
    accs  = [r["accuracy"] for r in results]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accs, color=colors, edgecolor="black")
    plt.ylim(0.8, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{acc:.4f}",
            ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()
    print("Model comparison plot saved → model_comparison.png")


def plot_confusion_matrix(best_result: dict, y_test) -> None:
    """Confusion matrix for the best model."""
    cm = confusion_matrix(y_test, best_result["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_result['name']}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved → confusion_matrix.png")


def plot_feature_importance(rf_model, feature_names: list) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="steelblue",
        edgecolor="black",
    )
    plt.xlabel("Feature Importance")
    plt.title("Random Forest — Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("Feature importance plot saved → feature_importance.png")


# ─────────────────────────────────────────
# 6. Main Pipeline
# ─────────────────────────────────────────
def main(filepath: str = "gym_churn.csv") -> None:
    # Load
    df = load_data(filepath)

    # EDA
    eda(df)

    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # Train & compare all models
    results = compare_models(X_train, X_test, y_train, y_test)

    # Pick best model
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n★ Best Model: {best['name']} (Accuracy: {best['accuracy']:.4f})")

    # Visualizations
    plot_model_comparison(results)
    plot_confusion_matrix(best, y_test)

    # Feature importance (Random Forest specific)
    rf_result = next(r for r in results if r["name"] == "Random Forest")
    feature_names = df.drop(columns=["Churn"]).columns.tolist()
    plot_feature_importance(rf_result["model"], feature_names)


if __name__ == "__main__":
    main("gym_churn.csv")
