# 🎓 Smart Student Performance Prediction System

A Machine Learning-based system that predicts student academic performance using demographic, academic, behavioral, and psychological data.

---

## 📌 Overview

This project builds a predictive analytics model to estimate student performance using a dataset of 20,000 samples and 38 features.

The system supports:
- Final score prediction (Regression)
- Final grade classification (A/B/C/D/F)
- Pass/Fail prediction

---

## 📊 Dataset

The dataset includes:

- Demographic data (age, gender, family background)
- Academic history (GPA, failed courses, study hours)
- Behavioral data (attendance, assignments, participation)
- Psychological factors (stress, motivation, sleep)
- Institutional data (course difficulty, class size)

---

## ⚙️ Pipeline

1. Data preprocessing (cleaning, encoding, scaling)
2. Exploratory Data Analysis (EDA)
3. Model training and testing
4. Performance evaluation

---

## 🧠 Models Used

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- Linear Regression  

---

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score  
- Confusion Matrix  
- ROC Curve & AUC  
- MSE, MAE, R²  

---

## 🛠️ Technologies

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 📁 Project Structure
Smart-Student-Performance-Prediction/
│
├── ML_Project_Finalll.ipynb      # Main notebook (EDA + models)
├── model.ipynb                  # Model experimentation
├── app.py                       # (Optional) application interface
├── Term_Project_Dataset_20K.csv # Dataset
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/Smart-Student-Performance-Prediction.git
cd Smart-Student-Performance-Prediction
pip install -r requirements.txt
jupyter notebook
