# Customer Churn Prediction — Short Report

## 1. Objective
The objective is to train a classifier that predicts whether a customer churned (`churn = 1`) or was retained (`churn = 0`) from customer behavior. The project implements a Decision Tree Classifier and compares it with Logistic Regression.

## 2. Dataset and preprocessing
The included CSV is a **synthetic educational dataset**, created to demonstrate a complete machine-learning workflow. It contains numerical customer features: age, purchase frequency, total spending, days since last purchase, support interactions, membership duration, and discount usage rate. Some numerical values are intentionally missing so imputation can be demonstrated.

The target is separated from the predictors. The data is split into 80% training and 20% testing records using stratification. Missing numeric values are imputed with the median. The imputer is placed inside each model's pipeline to avoid fitting preprocessing on test data. Standardization is applied to Logistic Regression; it is not required for a Decision Tree because tree splits depend on feature thresholds rather than feature distances.

## 3. Models
The Decision Tree uses Gini impurity, a maximum depth of 4, a minimum of 10 samples to split an internal node, a minimum of 5 samples per leaf, and balanced class weights. These settings constrain tree complexity and help reduce overfitting. Logistic Regression is included as a linear baseline and uses standard-scaled features.

## 4. Evaluation
Run `python src/train.py` to generate the metrics. The script reports accuracy, precision, recall, and F1-score for the churn class, plus confusion matrices for both models. The resulting values are saved in `reports/model_comparison.csv` and `reports/model_metrics.json`.

Do not copy numbers into this report before running the code. The dataset is synthetic, so these metrics only demonstrate the workflow; they do not establish expected performance on real customers.

For churn detection, recall is especially relevant because a false negative means the model failed to flag a customer who churned. Precision also matters because false positives can cause unnecessary retention offers. F1-score summarizes precision and recall as their harmonic mean.

## 5. Feature importance and tree interpretation
The program saves `reports/figures/feature_importance.png`, `reports/feature_importance.csv`, and `reports/figures/decision_tree.png`. Feature importance is based on the reduction in impurity attributed to each feature in this fitted tree. It is model-specific and should not be interpreted as proof of causation. Each tree node displays a split condition; follow the left/right branches to understand the model's rule, and read the leaf's predicted class.

## 6. Decision Tree versus Logistic Regression
Decision Trees are easy to visualize, can model nonlinear patterns and feature interactions, and generally do not require feature scaling. Their limitations include sensitivity to small changes in data, possible overfitting, and unstable impurity-based importance when predictors are correlated. Logistic Regression is a useful baseline and can be easier to interpret through coefficients, but its decision boundary is linear unless nonlinear features or interactions are added. Standardization helps its optimization.

## 7. Conclusion
This mini project demonstrates the full churn-classification workflow, including data cleaning, model training, evaluation, feature importance, tree visualization, and baseline comparison. Before using such a system in a real business, validate it on genuine, authorized customer data, check class imbalance and data leakage, tune thresholds based on business costs, and monitor performance over time.
