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

# 🚀 End-to-End Customer Churn Analytics & ML Modeling

> **Repository Focus:** Portfolio project tracking the complete machine learning lifecycle—from exploratory data analysis to advanced predictive modeling, cost-sensitive threshold tuning, and model interpretability.

---

## 📂 Project Structure & Milestones

* **`week-1-customer-churn-eda.ipynb`** → Exploratory data analysis, univariate/bivariate visualizations, correlation matrices, and data cleaning.
* **`week-2-building-ml-models.ipynb`** → Feature engineering, Scikit-Learn pipelines, baseline creation, model training (Logistic Regression, Decision Trees, Random Forests), evaluation metrics, and threshold optimization.

---

## 📈 Week 2: Key Modeling Results & Insights

### **1. Baseline vs. Advanced Models**
* **Baseline Model (DummyClassifier):** Achieved `0.735` accuracy by predicting all customers stay, but resulted in a **`0.000` recall**, demonstrating why accuracy is a misleading metric for imbalanced data.
* **Top Performers:** **Logistic Regression** and **Random Forest** achieved the highest overall performance with an **AUC of 0.842** and test accuracy of **0.807**.

### **2. Strategic Cost-Sensitive Thresholding**
* Instead of defaulting to the standard `0.5` classification cutoff, cost-matrix analysis dictated shifting the threshold down to **`0.15`**. 
* **Business Rationale:** Since missing a churner (False Negative) carries a much higher business cost than a False Alarm (False Positive), optimizing the threshold successfully maximizes true churn capture.

### **3. Class Imbalance & Feature Engineering**
* **Balanced Weights:** Utilizing `class_weight='balanced'` in Logistic Regression dramatically improved minority class recovery, lifting recall from `0.567` up to `0.781`.
* **Engineered Features:** Created customized service-count and contract-risk flags; while tree-based models natively handled these feature interactions, the AUC profile remained stable and robust.

---

## 🛠️ Tech Stack & Dependencies
* **Programming Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `imbalanced-learn`
* **Execution Environment:** Kaggle Notebooks

---

## 👤 Author
* **Mahum** (BS 7th Semester, Department of Electronics)  
* **Course:** Introduction to Applied AI | Instructor: Dr. Faiz Ahmed
