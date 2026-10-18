# Customer Churn Prediction: Decision Tree vs Logistic Regression

A beginner-friendly, GitHub-ready mini project for a Decision Tree assignment. It includes preprocessing, train/test splitting, evaluation metrics, confusion matrices, feature importance, tree visualization, and a Logistic Regression comparison.

> **Dataset notice:** `data/customer_churn_synthetic.csv` is a synthetic educational dataset generated for this project. It is not real customer data, and its model scores must not be represented as real-world business performance. Replace it with an approved real dataset before making business claims.

## Project structure

```text
customer_churn_decision_tree/
├── data/
│   └── customer_churn_synthetic.csv
├── notebooks/
│   └── customer_churn_decision_tree.ipynb
├── reports/
│   ├── report.md
│   └── figures/                 # generated after running the pipeline
├── src/
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 1. Setup (Windows PowerShell)

```powershell
# Open the project directory
cd customer_churn_decision_tree

# Optional: create and activate a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and run the script with `.\.venv\Scripts\python.exe src\train.py`.

## 2. Run the project

```powershell
python src/train.py
```

The script will:
- Load the CSV and validate the target column.
- Separate features (`X`) from target (`y`).
- Split the data into train and test sets using stratification.
- Impute missing numerical values using the **training set median**.
- Train a Decision Tree and Logistic Regression using a scikit-learn pipeline.
- Print accuracy, precision, recall, F1-score, and confusion matrices.
- Save confusion matrix plots, feature importance, and the decision-tree visualization.
- Save a machine-readable metrics file to `reports/model_metrics.json`.

Generated files are written to `reports/figures/`.

## 3. Open the notebook

Open `notebooks/customer_churn_decision_tree.ipynb` in VS Code or Jupyter and run the cells from top to bottom. The notebook walks through the same concepts in smaller steps.

## 4. What the target means

- `churn = 1`: the customer churned.
- `churn = 0`: the customer was retained.

For churn prediction, **recall for class 1** is important because a false negative is a churned customer that the model failed to flag. Precision also matters because false positives can lead to unnecessary retention offers. The best trade-off depends on business costs.

## 5. Why scaling is not required for a Decision Tree

Decision trees split a feature at thresholds and do not depend on distances between feature values, so scaling is generally unnecessary. Logistic Regression is sensitive to feature scales, so this project uses `StandardScaler` in the Logistic Regression pipeline only. This keeps preprocessing appropriate for each model.

## 6. Avoiding data leakage

The imputer and scaler are inside scikit-learn pipelines. They are fitted only on the training set, then applied to the test set. The test set is used for final evaluation, not for model selection.

## 7. Replacing the dataset

Use a CSV with one row per customer and a binary `churn` column (0 = retained, 1 = churned). Update `DATA_PATH` in `src/train.py` if the file has a different name. Keep only features that would genuinely be available at prediction time. For example, do not use a field recorded only after a customer has already churned.

## 8. Assignment mapping

| Assignment requirement | Implemented in |
|---|---|
| a) Load data, handle missing values, split data | `src/train.py`, notebook |
| b) Train `DecisionTreeClassifier` | `src/train.py`, notebook |
| c) Metrics and confusion matrix | `src/train.py`, notebook |
| d) Feature importance chart | `src/train.py`, notebook |
| e) Tree plot and interpretation | `src/train.py`, notebook |
| f) Compare Logistic Regression and discuss trade-offs | `src/train.py`, report |

## 9. GitHub publishing

```powershell
git init
git add .
git commit -m "Build customer churn prediction mini project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-decision-tree.git
git push -u origin main
```

Create the repository on GitHub first, then replace `YOUR_USERNAME` with your GitHub username. Do not commit virtual environments, secrets, or private customer data.
