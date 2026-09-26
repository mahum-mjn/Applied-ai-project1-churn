# Customer-churn-prediction
Customer churn prediction using machine learning

## Week 1: Exploratory Data Analysis 
 
### Dataset - Source: Telco Customer Churn (Kaggle) - Size: 7,043 customers, 21 features - Target: Predict customer churn (Yes/No) 
 
### Key Findings 
1. Short tenure combined with month-to-month contracts represents the primary indicator of immediate customer loss.
2. High monthly charges drive churn unless anchored by long-term contract discounts or bundle retention offers.
3. Fiber optic customers churn at a higher rate despite paying more, pointing to potential service quality or pricing issues. 

 ## Setup
* **Live Interactive Notebook:** [Run on Kaggle](https://www.kaggle.com/code/mahumjunaid/week-1-customer-churn-eda)
* **Local Setup:** `pip install pandas numpy matplotlib seaborn`

 # 📊 Customer Churn Prediction & ML Model Evaluation (Lab 2)

## **Project Overview**
This project explores end-to-end machine learning workflows for customer churn prediction using a telecommunications dataset. Built as part of the *Introduction to Applied AI* course, this notebook covers everything from rigorous data preprocessing and pipeline construction to advanced model tuning, cost-sensitive threshold optimization, and model interpretability.

---

## **Key Highlights & Results**
- **Baseline Model:** Achieved an accuracy of `0.735`, but yielded a recall of `0.000`, proving that raw accuracy fails on imbalanced data.
- **Best Models:** **Logistic Regression** and **Random Forest** tied for the highest test performance with an **AUC of 0.842** and an **accuracy of 0.807**.
- **Cost-Optimized Threshold:** Shifted the classification threshold from the default `0.5` down to **`0.15`** based on business cost trade-offs (where a False Negative is significantly more costly than a False Positive).
- **Class Imbalance:** Evaluated `class_weight='balanced'` in Logistic Regression, boosting recall from `0.567` up to `0.781`.
- **Feature Engineering:** Tested custom domain features (service counts, tenure flags, price jumps), discovering that tree ensembles natively capture these interactions without inflating test AUC.

---

## **Tech Stack & Libraries**
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib, Imbalanced-Learn
* **Platform:** Kaggle Notebook

---

## **Author**
* **Mahum** (BS 7th Semester, Department of Electronics)
* **Course:** Introduction to Applied AI (Instructor: Dr. Faiz Ahmed)
