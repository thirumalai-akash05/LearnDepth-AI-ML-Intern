# Student Pass/Fail Analysis

## 📌 Project Overview

This project analyzes student academic data to understand which factors are associated with student **Pass/Fail outcomes**.

The analysis focuses on study hours, attendance, assignment completion, and previous scores. The dataset is inspected, cleaned, visualized, and prepared for basic machine learning.

## 🎯 Objectives

* Load and inspect the student academic dataset
* Identify rows, columns, and data types
* Detect missing values and duplicate records
* Identify and fix invalid values
* Detect potential outliers using boxplots
* Analyze the distribution of study hours
* Compare Pass vs Fail outcomes
* Analyze study hours and attendance in relation to student outcomes
* Create a correlation matrix and heatmap
* Perform basic data cleaning
* Create new features
* Separate features (`X`) and target (`y`) for machine learning

## 📊 Dataset

The dataset contains the following columns:

| Column                  | Description                             |
| ----------------------- | --------------------------------------- |
| `study_hours`           | Number of hours spent studying          |
| `attendance`            | Student attendance percentage           |
| `assignments_completed` | Number of assignments completed         |
| `previous_score`        | Student's previous academic score       |
| `pass`                  | Target variable: `0 = Fail`, `1 = Pass` |

## 🔍 Data Quality Checks

The dataset was checked for:

* Missing values
* Duplicate rows
* Negative study hours
* Attendance values above 100%
* Negative assignment values
* Previous scores above 100
* Potential outliers

Invalid numerical values were replaced with missing values and subsequently handled using median imputation.

## 🧹 Data Cleaning

The following preprocessing steps were performed:

1. Removed duplicate rows
2. Identified invalid numerical values
3. Replaced invalid values with `NaN`
4. Filled missing numerical values using the median
5. Verified that the cleaned dataset contains no missing values or duplicates

## 🛠️ Feature Engineering

Two new features were created:

### 1. `academic_effort`

Combines study hours and assignments completed to represent a basic measure of academic effort.

### 2. `attendance_good`

A binary feature based on attendance:

* `1` → Attendance is 75% or above
* `0` → Attendance is below 75%

## 📈 Data Visualization

The project includes visualizations for:

* Study hours distribution
* Pass vs Fail distribution
* Study hours vs Pass/Fail
* Attendance vs Pass/Fail
* Numerical feature boxplots
* Correlation matrix / heatmap

## 📂 Project Structure

```text
student-pass-fail-analysis/
│
├── 01_student_pass_fail.csv
├── student_pass_fail_analysis.ipynb
├── cleaned_student_pass_fail.csv
│
├── README1.md
```

## 💻 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook
* VS Code

## 📌 Machine Learning Preparation

After cleaning and feature engineering, the dataset is separated into:

**Features (`X`):**

* `study_hours`
* `attendance`
* `assignments_completed`
* `previous_score`
* `academic_effort`
* `attendance_good`

**Target (`y`):**

* `pass`

The prepared dataset can be used for basic classification algorithms in future work.

## 📊 Key Learning Outcomes

Through this project, I learned how to:

* Inspect a messy numerical dataset
* Identify common data-quality problems
* Handle missing and invalid values
* Remove duplicate records
* Detect potential outliers
* Create meaningful visualizations
* Understand basic correlations
* Perform simple feature engineering
* Prepare a dataset for machine learning

## 🚀 Future Improvements

* Apply classification algorithms such as Logistic Regression, Decision Tree, and Random Forest
* Split the data into training and testing sets
* Evaluate model performance using accuracy, precision, recall, and F1-score
* Compare multiple machine learning models
* Perform additional feature analysis

## 👨‍💻 Author

**Thirumalai Akash M**

B.Tech Information Technology Student

---

⭐ If you find this project useful, feel free to explore the notebook and analysis.
