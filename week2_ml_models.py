"""
Week 2 Task: Supervised Machine Learning Models
Student: Binnala Esikha

This demonstration uses an illustrative sample dataset, not real student records.
Run: python week2_ml_models.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_FILE = "student_performance.csv"

def main():
    data = pd.read_csv(DATA_FILE)
    features = [
        "hours_studied_per_week",
        "attendance_percent",
        "previous_score",
        "extracurricular_participation",
    ]
    X = data[features]
    y_reg = data["final_score"]
    y_clf = data["passed"]

    X_train, X_test, yreg_train, yreg_test = train_test_split(
        X, y_reg, test_size=0.25, random_state=42
    )
    # Keep classification split consistent for a fair model comparison.
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_clf, test_size=0.25, random_state=42, stratify=y_clf
    )

    print("\\n=== REGRESSION: predict final score ===")
    reg = LinearRegression()
    reg.fit(X_train, yreg_train)
    reg_pred = reg.predict(X_test)
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(yreg_test, reg_pred):.2f}")
    print(f"Root Mean Squared Error (RMSE): {mean_squared_error(yreg_test, reg_pred) ** 0.5:.2f}")
    print(f"R-squared (R2): {r2_score(yreg_test, reg_pred):.3f}")

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }

    print("\\n=== CLASSIFICATION: predict pass/fail ===")
    results = []
    for name, model in models.items():
        model.fit(Xc_train, yc_train)
        pred = model.predict(Xc_test)
        scores = {
            "Model": name,
            "Accuracy": accuracy_score(yc_test, pred),
            "Precision": precision_score(yc_test, pred, zero_division=0),
            "Recall": recall_score(yc_test, pred, zero_division=0),
            "F1-score": f1_score(yc_test, pred, zero_division=0),
        }
        results.append(scores)
        print(
            f"{name}: Accuracy={scores['Accuracy']:.3f}, "
            f"Precision={scores['Precision']:.3f}, "
            f"Recall={scores['Recall']:.3f}, F1={scores['F1-score']:.3f}"
        )

    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    results_df.to_csv("classification_model_comparison.csv", index=False)
    print("\\nComparison saved to classification_model_comparison.csv")
    print("\\nNote: this is a small illustrative dataset. Metrics are for demonstration only.")

if __name__ == "__main__":
    main()
