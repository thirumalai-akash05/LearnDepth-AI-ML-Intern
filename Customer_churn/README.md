# Customer Churn Risk Prediction

## 📌 Project Overview

Customer churn is an important business problem where companies need to identify customers who are likely to stop using their subscription services.

This project develops a **Binary Classification Machine Learning model** to predict whether a subscription customer is at risk of churning in the next billing cycle.

The project uses **Logistic Regression** as the primary classification algorithm. The model analyzes customer behavior and subscription-related features to classify customers into:

* `0` → No Churn Risk
* `1` → Churn Risk

The project follows a complete machine learning workflow, including data inspection, preprocessing, train-test splitting, model training, evaluation, and feature interpretation.

---

## 🎯 Objective

The main objective of this project is to build a reproducible machine learning solution that can:

* Analyze customer subscription data
* Identify factors associated with churn risk
* Predict whether a customer belongs to the churn-risk class
* Evaluate the model using multiple classification metrics
* Interpret the importance and direction of model coefficients

---

## 📊 Dataset

The dataset contains **1,000 customer records** with **6 predictor variables** and **1 target variable**.

### Features

| Feature               | Description                                       |
| --------------------- | ------------------------------------------------- |
| `tenure_months`       | Number of months the customer has been subscribed |
| `monthly_charges`     | Customer's monthly subscription charges           |
| `support_tickets`     | Number of support tickets raised by the customer  |
| `avg_session_minutes` | Average session duration in minutes               |
| `late_payments`       | Number of late payments                           |
| `contract_months`     | Contract duration in months                       |
| `target`              | Churn-risk target: `0` = No Risk, `1` = Risk      |

### Dataset Quality

* Total records: **1,000**
* Predictor variables: **6**
* Target variable: **1**
* Missing values: **0**
* Duplicate rows: **0**
* Class 0: **500 records**
* Class 1: **500 records**
* Class distribution: **50% / 50%**

The target is therefore balanced, so no class-balancing technique was required for this experiment.

---

## 🔄 Machine Learning Workflow

```text
Customer Churn Dataset
        ↓
Data Inspection
        ↓
Data Quality Checking
        ↓
Exploratory Data Analysis
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Logistic Regression
        ↓
Prediction
        ↓
Model Evaluation
        ↓
Confusion Matrix & ROC-AUC
        ↓
Coefficient Interpretation
        ↓
Conclusion & Limitations
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Jupyter Notebook**

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Checked the dataset structure and data types.
2. Checked for missing values.
3. Checked for duplicate records.
4. Examined the target class distribution.
5. Separated predictors (`X`) and target (`y`).
6. Divided the dataset into training and testing sets.
7. Applied `StandardScaler` to standardize the numerical features.

An **80:20 train-test split** was used with:

```text
Test Size = 20%
Random State = 42
Stratification = Target Variable
```

Stratification was used to maintain the same class distribution in both training and testing datasets.

### Leakage Prevention

Feature scaling was implemented inside a **Scikit-learn Pipeline**.

This ensures that the scaler is fitted only on the training data and prevents information from the test set from leaking into the training process.

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression was selected as the primary model because the problem is a **binary classification task**.

The model estimates the probability that a customer belongs to the churn-risk class.

The basic logistic regression model can be represented as:

```text
P(Y = 1) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))
```

The predicted probability is converted into a class prediction using a classification threshold.

### Model Configuration

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])
```

---

## 📈 Model Evaluation

The model was evaluated using the required classification metrics.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **68.00%** |
| Precision | **68.37%** |
| Recall    | **67.00%** |
| F1-Score  | **67.68%** |
| ROC-AUC   | **74.95%** |

### Interpretation

* **Accuracy – 68.00%:** The model correctly classified 68% of the test customers.
* **Precision – 68.37%:** Around 68% of customers predicted as churn-risk customers were actually in the positive class.
* **Recall – 67.00%:** The model identified 67% of the actual positive-class customers.
* **F1-Score – 67.68%:** Provides a balance between precision and recall.
* **ROC-AUC – 74.95%:** Indicates that the model has reasonable ability to distinguish between the two classes.

---

## 🔲 Confusion Matrix

The confusion matrix obtained from the test set was:

```text
                 Predicted
                 0       1

Actual 0        69      31
Actual 1        33      67
```

### Components

* **True Negative (TN): 69**
* **False Positive (FP): 31**
* **False Negative (FN): 33**
* **True Positive (TP): 67**

The model correctly classified **69 negative-class customers** and **67 positive-class customers**.

There were **31 false positives** and **33 false negatives**.

For a churn-risk application, false negatives are particularly important because they represent customers who may be at risk but were not identified by the model.

---

## 🔍 Feature / Coefficient Interpretation

Because the features were standardized, the coefficients represent the change in the model's log-odds associated with approximately a **one-standard-deviation increase** in each feature, while holding the other features constant.

| Feature               | Coefficient | Direction |
| --------------------- | ----------: | --------- |
| `monthly_charges`     |      +0.647 | Positive  |
| `contract_months`     |      +0.536 | Positive  |
| `avg_session_minutes` |      +0.508 | Positive  |
| `support_tickets`     |      -0.448 | Negative  |
| `late_payments`       |      -0.428 | Negative  |
| `tenure_months`       |      -0.351 | Negative  |

### Key Observations

**Monthly charges** has the strongest positive coefficient. In this dataset, higher monthly charges are associated with a higher predicted probability of the positive class.

**Contract duration** also has a positive coefficient. This is a dataset-specific relationship and should not automatically be interpreted as a real-world causal relationship.

**Average session minutes** has a positive relationship with the predicted positive class.

**Support tickets**, **late payments**, and **tenure** have negative coefficients in this dataset.

> Important: These coefficients represent statistical associations learned by the model. They should not be interpreted as proof that a feature causes or prevents customer churn.

---

## 🧪 Baseline Comparison

A simple `DummyClassifier` baseline was also considered.

| Model               |   Accuracy |    ROC-AUC |
| ------------------- | ---------: | ---------: |
| Dummy Baseline      |     50.00% |     50.00% |
| Logistic Regression | **68.00%** | **74.95%** |

The Logistic Regression model performs substantially better than the naive baseline, indicating that the selected customer features contain useful predictive information.

---

## 📁 Project Structure

```text
customer-churn-risk-prediction/
│
├── README.md
├── customer_churn_prediction.ipynb
├── requirements.txt
│
├── dataset/
│   └── dataset_01_customer_churn_risk.csv
│
├── report/
│   └── Customer_Churn_Risk_Report.pdf
│
└── images/
    ├── confusion_matrix.png
    └── roc_curve.png
```

> If the dataset is not permitted to be publicly shared, remove it from the repository and provide instructions for obtaining it separately.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-risk-prediction.git
```

### 2. Navigate to the project directory

```bash
cd customer-churn-risk-prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

Open:

```text
customer_churn_prediction.ipynb
```

Run the cells sequentially to reproduce the analysis and model results.

---

## 📦 Requirements

The main Python libraries used in this project are:

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

---

## ⚠️ Limitations

Although the model provides useful results, there are several limitations:

1. The dataset contains only **1,000 records**.
2. Only **six predictors** are available.
3. Statistical outliers are present in some variables, particularly `monthly_charges`.
4. The evaluation uses a single train-test split rather than cross-validation.
5. The model identifies associations and does not establish causation.
6. The default probability threshold of `0.5` may not be optimal for a real-world churn-retention strategy.
7. Real customer churn may depend on additional factors such as customer satisfaction, service quality, complaints, discounts, pricing changes, competitor offers, and customer demographics.

---

## 🔮 Future Improvements

The project can be improved by:

* Using a larger and more diverse customer dataset
* Adding additional customer behavior and subscription features
* Applying k-fold cross-validation
* Performing hyperparameter tuning
* Testing additional classification algorithms
* Optimizing the classification threshold
* Handling influential outliers more carefully
* Applying feature selection techniques
* Monitoring model performance over time
* Deploying the model as a web application or API
* Building a customer churn-risk dashboard

---

## 📌 Conclusion

This project demonstrates a complete machine learning workflow for predicting customer churn risk using Logistic Regression.

The model achieved:

```text
Accuracy  : 68.00%
Precision : 68.37%
Recall    : 67.00%
F1-Score  : 67.68%
ROC-AUC   : 74.95%
```

The ROC-AUC score of **74.95%** indicates reasonable discrimination between the two target classes.

The project also demonstrates the importance of proper preprocessing, stratified train-test splitting, leakage prevention, evaluation using multiple metrics, confusion-matrix analysis, and interpretation of model coefficients.

While the current model is useful as a baseline, further data, validation, feature engineering, and threshold optimization would be required before considering deployment in a real business environment.

---

## 👨‍💻 Author

**Thirumalai Akash M**

B.Tech Information Technology
Mepco Schlenk Engineering College

### Areas of Interest

* Python
* Machine Learning
* Artificial Intelligence
* Data Science
* Data Analytics

---

## ⭐ Project Highlights

* ✅ Binary Classification
* ✅ Logistic Regression
* ✅ StandardScaler
* ✅ Scikit-learn Pipeline
* ✅ Stratified Train-Test Split
* ✅ Accuracy, Precision, Recall & F1-Score
* ✅ ROC-AUC Analysis
* ✅ Confusion Matrix
* ✅ Feature Coefficient Interpretation
* ✅ Baseline Comparison
* ✅ Reproducible Machine Learning Workflow
