# Week 2: Supervised Machine Learning Models

**Student:** Binnala Esikha  
**Internship:** YuvaIntern — AI Pioneers Internship  
**Task:** Build regression and classification models using Scikit-learn.

## Objective
Train supervised learning models, make predictions, and compare performance using evaluation metrics.

## Models included
- Linear Regression — predicts a numerical final score.
- Logistic Regression — classifies pass/fail.
- Decision Tree Classifier — classifies pass/fail using decision rules.
- Random Forest Classifier — combines multiple decision trees.
- K-Nearest Neighbors (KNN) — classifies using nearby examples.

## Files
- `week2_ml_models.py` — complete training, prediction, and evaluation script.
- `student_performance.csv` — illustrative sample dataset.
- `requirements.txt` — required Python packages.
- `Week2_Report.docx` — project report.
- `classification_model_comparison.csv` — generated when the script runs.

## Dataset
The CSV contains 40 illustrative rows with study hours, attendance, previous score, extracurricular participation, final score, and a pass/fail label. It is sample data created for learning and is **not real student data**.

## Run
Install packages:
```bash
pip install -r requirements.txt
```
Place the CSV and Python file in the same folder, then run:
```bash
python week2_ml_models.py
```

The script prints regression and classification metrics and saves the classification comparison to a CSV file.

## Evaluation
Regression metrics: MAE, RMSE, and R².  
Classification metrics: accuracy, precision, recall, and F1-score.

## Conclusion
This project demonstrates the end-to-end supervised learning workflow: loading data, separating features and targets, splitting training/testing data, training models, generating predictions, evaluating performance, and comparing classifiers. Results from this small sample dataset are educational only and should not be treated as evidence of real-world predictive performance.
