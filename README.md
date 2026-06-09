[README (1).md](https://github.com/user-attachments/files/28770302/README.1.md)
# 🏋️ Gym Membership Churn Prediction

> A machine learning project that predicts whether a gym member will cancel their membership.  
> **Group 5** — Wei Han Lee · Kittinat Tongmai · Jihye Kim

---

## 📌 Problem Statement

Gym membership churn is a major challenge for fitness businesses. This project builds a binary classification model to predict whether a member will churn (`1`) or stay (`0`), enabling the gym to take proactive retention actions.

---

## 📂 Dataset

| Feature | Description |
|---|---|
| `gender` | Gender (0 = Female, 1 = Male) |
| `Near_Location` | Gym near member's home (1 = Yes, 0 = No) |
| `Partner` | Has a partner membership (1 = Yes, 0 = No) |
| `Promo_friends` | Joined through friend promotion (1 = Yes, 0 = No) |
| `Phone` | Provided phone number (1 = Yes, 0 = No) |
| `Contract_period` | Contract duration (months) |
| `Group_visits` | Attends group classes (1 = Yes, 0 = No) |
| `Age` | Member age |
| `Avg_additional_charges_total` | Average extra charges |
| `Month_to_end_contract` | Months remaining on contract |
| `Lifetime` | Membership duration (months) |
| `Avg_class_frequency_total` | Overall class attendance frequency |
| `Avg_class_frequency_current_month` | This month's attendance frequency |
| `Churn` | **Target** — Churned (1 = Yes, 0 = No) |

### Dataset Issues & Preprocessing

- **Class Imbalance** → handled with **SMOTE** (oversampling minority class on training set only)
- **Missing Values** → rows with null values dropped
- **Feature Scaling** → all features standardized with `StandardScaler`

---

## 🤖 Models Compared

| Model | Pros | Cons |
|---|---|---|
| Logistic Regression | Simple, interpretable, fast | May miss complex patterns |
| **Random Forest** ✅ | Robust, handles non-linearity, feature importance | Less interpretable |
| SVM | Good in high-dim spaces | Slow, hard to interpret |
| Gradient Boosting | Often high accuracy | Prone to overfitting, slow |

### Why Random Forest?

- Provides **feature importance** to understand key churn drivers
- Naturally **robust to overfitting** by averaging multiple trees
- Captures **non-linear relationships** between features
- Works well with **minimal preprocessing**

---

## 📊 Results

The Random Forest model achieved:

| Metric | Class 0 (No Churn) | Class 1 (Churn) |
|---|---|---|
| Precision | 0.95 | 0.93 |
| Recall | 0.93 | 0.95 |
| F1-Score | 0.94 | 0.94 |

**Overall Accuracy: 94.13%**

Both classes show balanced precision and recall, confirming the model generalizes well to unseen data.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/gym-churn-prediction.git
cd gym-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place your `gym_churn.csv` file in the project root directory.  
The dataset can be downloaded from [Kaggle — Gym Members Exercise Dataset](https://www.kaggle.com/) or your course materials.

### 4. Run the pipeline

```bash
python gym_churn_prediction.py
```

This will:
1. Load and run EDA on the dataset
2. Preprocess data (SMOTE + StandardScaler)
3. Train and compare all four models
4. Print classification reports
5. Save plots: `eda_overview.png`, `model_comparison.png`, `confusion_matrix.png`, `feature_importance.png`

---

## 📁 Project Structure

```
gym-churn-prediction/
├── gym_churn_prediction.py   # Main ML pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── gym_churn.csv             # Dataset (add manually)
```

---

## 🔍 Output Plots

| File | Description |
|---|---|
| `eda_overview.png` | Class distribution + correlation heatmap |
| `model_comparison.png` | Accuracy comparison across 4 models |
| `confusion_matrix.png` | Confusion matrix for best model |
| `feature_importance.png` | Top features driving churn prediction |

---

## 📚 Libraries Used

- [scikit-learn](https://scikit-learn.org/) — ML models & metrics
- [imbalanced-learn](https://imbalanced-learn.org/) — SMOTE
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — Data processing
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) — Visualization

---

## 👥 Team

Wei Han Lee · Kittinat Tongmai · Jihye Kim
