# Applied AI Project 1: Customer Churn Prediction & Optimization

# Customer-churn-prediction
Customer churn prediction using machine learning

## 🔗 Live Interactive Notebooks
* **Week 1 EDA Notebook:** [View & Run on Kaggle](https://www.kaggle.com/code/mahumjunaid/week-1-customer-churn-eda)
* **Week 2 Modeling Notebook:** [View & Run on Kaggle](https://www.kaggle.com/code/mahumjunaid/week-2-building-ml-models?scriptVersionId=352934988)
* **Week 3 Optimization Notebook:** [Local/Kaggle Execution (`week3-optimization.ipynb`)]
---

## 📂 Project Structure & Milestones
* **`week1-eda.ipynb`** → Exploratory data analysis, data cleaning, and contract-specific churn distribution profiling.
* **`week2-ml-models.ipynb`** → Scikit-Learn pipelines, baseline benchmarking, logistic regression odds ratios, tree ensembles, and cost-optimized threshold tuning.
* **`week3-optimization.ipynb`** → Cross-validation stability analysis, hyperparameter tuning, XGBoost early stopping, K-Means customer segmentation, and PCA dimensionality reduction.
* **`churn_model.joblib`** → Serialized production-ready XGBoost classification model.

---

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

### **Week 2: Machine Learning Modeling & Threshold Optimization**
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

# Week 3: Model Optimization, Unsupervised Learning & Dimensionality Reduction
* **Split-to-Split Sensitivity & Cross-Validation:** 
  * Evaluating models across 20 random seeds revealed a wide accuracy variation ranging from **79.2% to 83.1%**, proving that a single random train/test split is highly unstable.
  * Switched to 5-fold cross-validation, achieving robust performance:
    * **Logistic Regression (Tuned C):** $0.8464 \pm 0.0129$ AUC
    * **Random Forest (Random Search):** $0.8464 \pm 0.0114$ AUC
    * **XGBoost (Tuned):** $0.8502 \pm 0.0117$ AUC
* **Hyperparameter Optimization & XGBoost:** 
  * Randomized Search efficiently identified optimal tree parameters. Early stopping on XGBoost successfully prevented overfitting by selecting **247 trees**.
  * The final model achieved a test AUC of **0.8483**, falling comfortably within the expected $\pm 2$ standard deviation CV range ($[0.8268, 0.8736]$). The final model has been serialized as `churn_model.joblib`.
* **Unsupervised Customer Segmentation ($k = 4$):** 
  * Using scaled features and silhouette analysis, $k = 4$ was selected. Post-hoc profiling revealed key actionable groups:
    * **At-Risk New High-Spenders:** 43% churn rate *(Action: Immediate onboarding & bundle discounts)*.
    * **New Budget / Low-Engagement:** 32% churn rate *(Action: Cross-sell add-ons at promo rates)*.
    * **Loyal High-Value VIPs:** 14% churn rate *(Action: Enroll in VIP/referral programs)*.
    * **Stable Long-Term Budget:** 5% churn rate *(Action: Maintain automated self-service)*.
* **Dimensionality Reduction (PCA):** Exactly **15 out of 30 components** explain 90% of cumulative variance. PC1 loadings highlighted severe multi-collinearity in "No internet service" dummy variables, while 2D scatter plots showed heavy class overlap.
* **Biggest Lesson:** Gradient boosting combined with rigorous 5-fold cross-validation is essential to prevent overfitting to a single random split and capture true generalization patterns.

---

## 🛠️ Tech Stack & Dependencies
* **Programming Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `imbalanced-learn`
* **Environment:** Kaggle Notebooks

---

