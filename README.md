# 🧠 Employee Salary Prediction (<=50K or >50K)

This is a Machine Learning web application built using Python and Streamlit to predict whether an employee's salary is greater than 50K or less than or equal to 50K per year, based on various input features like education, work hours, occupation, etc.

## 📌 Project Overview

- 🎯 Goal: Predict an employee's salary category using classification algorithms.
- 📊 Input: Employee-related features such as age, workclass, education, capital gain/loss, hours per week, etc.
- 🔍 Output: Predicted salary class - either `<=50K` or `>50K`.
- 🌐 Deployed using: Streamlit
- 📁 Dataset Source: Provided by internship organization (custom dataset)

---

## 🚀 Features

- Upload and load dataset for training & visualization.
- Clean and preprocess the data.
- Train a machine learning classification model.
- Use Streamlit web app for predictions based on manual user input.
- Display predicted label instantly.
- Interactive UI built with Streamlit widgets.
- Plots and performance metrics (accuracy, confusion matrix).

---

## 🛠️ Tech Stack and Libraries Used

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Streamlit | Front-end web UI for ML deployment |
| Pandas | Data handling and manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine learning modeling and evaluation |

---

## 🧮 How the Model Works

The model uses Logistic Regression or another classifier from scikit-learn. The process involves:

1. Data Cleaning
2. Label Encoding
3. Splitting data into training/testing sets
4. Model training
5. Prediction
6. Deployment in Streamlit

---

## 🧾 Dataset Columns

Sample columns from the dataset:
- age
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country
- salary (target)

---

## 🧪 How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction
