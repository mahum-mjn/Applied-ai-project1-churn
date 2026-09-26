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

# End-to-End Customer Churn Analytics & ML Modeling

> **Repository Focus:** Complete machine learning lifecycle portfolio tracking exploratory data analysis, feature engineering, pipeline implementation, cost-sensitive model evaluation, and threshold optimization.

---

## 🔗 Live Interactive Notebooks
* **Week 1 EDA Notebook:** [View & Run on Kaggle]((https://www.kaggle.com/code/mahumjunaid/week-1-customer-churn-eda))
* **Week 2 Modeling Notebook:** [View & Run on Kaggle](https://www.kaggle.com/code/mahumjunaid/week-2-building-ml-models/edit)

---

## 📂 Project Structure & Milestones
* **`week-1-customer-churn-eda.ipynb`** → Exploratory data analysis, data cleaning, and contract-specific churn distribution profiling.
* **`week-2-building-ml-models.ipynb`** → Scikit-Learn pipelines, baseline benchmarking, logistic regression odds ratios, tree ensembles, and cost-optimized threshold tuning.

---

## 📊 Phase Summary & Key Findings

### **Week 1: Exploratory Data Analysis Insights**
* **Dataset Overview:** Telco Customer Churn dataset comprising `7,043` customers and `21` features, targeting binary churn prediction.
* **Contract-Type Churn Breakdown:** Churn rates vary severely by contract type—**Month-to-month contracts** experience a high churn rate of `~42.7%`, whereas **One-year contracts** drop to `~11.3%`, and **Two-year contracts** plummet to `~2.8%`.
* **Service Drivers:** Fiber optic internet users exhibit a disproportionately high churn rate (`~41.9%` compared to DSL at `~19.5%`), highlighting structural pricing or service quality friction.

### **Week 2: Machine Learning Modeling & Threshold Optimization**
* **Baseline vs. Advanced Performance:** A dummy baseline achieved `73.5%` accuracy but yielded a `0.000` recall, proving that raw accuracy is deceptive on imbalanced datasets (~26.5% positive churn class). **Logistic Regression** and **Random Forest** achieved top-tier performance, both hitting an **AUC of 0.842** and test accuracy of **0.807**.
* **Class Imbalance Mitigation:** Utilizing `class_weight='balanced'` in Logistic Regression lifted minority class recovery, raising recall from `56.7%` to `78.1%`.
* **Strategic Threshold Tuning:** Incorporating business cost constraints (where missing a churner costs 4x more than a false alarm) justified shifting the classification threshold from the default `0.5` down to an optimal **`0.15`**, maximizing true customer retention capture.

---

## 🛠️ Tech Stack & Dependencies
* **Programming Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `imbalanced-learn`
* **Environment:** Kaggle Notebooks

---

## 👤 Author
* **Mahum** (BS 7th Semester, Department of Electronics)  
* **Course:** Introduction to Applied AI | Instructor: Dr. Faiz Ahmed[cite: 6]
