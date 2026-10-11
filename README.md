# 📊 Applied AI Project 1: Customer Churn Prediction, Optimization & Deployment

An end-to-end machine learning and applied AI project tracking the complete lifecycle of customer churn prediction—from exploratory data analysis to production-ready web application deployment.

---

## 🔗 Live Links & Interactive Notebooks
- **Week 1 EDA Notebook:**  [View & Run on Kaggle](https://www.kaggle.com/code/mahumjunaid/week-1-customer-churn-eda)
- **Week 2 Modeling Notebook:** [View & Run on Kaggle](https://www.kaggle.com/code/mahumjunaid/week-2-building-ml-models?scriptVersionId=352934988)
- **Week 3 Optimization Notebook:**  [View & Run on Kaggle ([`week3-optimization.ipynb`]([https://www.kaggle.com/code/mahumjunaid/week-3-model-optimization#Lab-3:-Model-Optimization,-Unsupervised-Learning-&-Dimensionality-Reduction](https://www.kaggle.com/code/mahumjunaid/week-3-model-optimization/notebook))]
- **Week 4 Deployment App:** [Try the Live Web App](https://churn-risk-advisor-inrvjcl5wdsnferw2nziyx.streamlit.app)
- **Production Repository:** [mahum-mjn/churn-risk-advisor](https://github.com/mahum-mjn/churn-risk-advisor)

---

## 📂 Project Structure & Milestones

- **`week1-eda.ipynb`** → Exploratory data analysis, data cleaning, and contract-specific churn distribution profiling.
- **`week2-ml-models.ipynb`** → Scikit-Learn pipelines, baseline benchmarking, logistic regression odds ratios, tree ensembles, and cost-optimized threshold tuning.
- **`week3-optimization.ipynb`** → Cross-validation stability analysis, hyperparameter tuning, XGBoost early stopping, K-Means customer segmentation, PCA dimensionality reduction, and updated preprocessing hand-offs for Week 4 deployment.
- **`churn_model.joblib`** → Serialized production-ready XGBoost classification model.
- **`app.py`** → Interactive Streamlit decision-support application with real-time "What-If" scenario testing.

---

## 📈 Weekly Evolution & Key Results

### 🔹 Week 1: Exploratory Data Analysis
**Dataset Source:** Telco Customer Churn (Kaggle) | **Size:** 7,043 customers, 21 features | **Target:** Customer Churn (Yes/No)

#### Key Findings
1. **Tenure & Contracts:** Short tenure combined with month-to-month contracts represents the primary indicator of immediate customer loss.
2. **Pricing Impact:** High monthly charges drive churn unless anchored by long-term contract discounts or bundle retention offers.
3. **Service Quality:** Fiber optic customers churn at a higher rate despite paying more, pointing to potential service quality or pricing sensitivity issues.

---

### 🔹 Week 2: Machine Learning Modeling & Threshold Optimization

#### Key Insights
1. **Baseline vs. Advanced Models:**
   - **Baseline (DummyClassifier):** Achieved `0.735` accuracy by predicting all customers stay, but resulted in `0.000` recall, demonstrating why accuracy is misleading for imbalanced data.
   - **Top Performers:** Logistic Regression and Random Forest achieved the highest overall performance with an **AUC of 0.842** and test accuracy of **0.807**.
2. **Strategic Cost-Sensitive Thresholding:**
   - Instead of defaulting to standard `0.50`, cost-matrix analysis dictated shifting the decision threshold down to **`0.15`–`0.30`**.
   - **Business Rationale:** Missing a churner (False Negative) carries a much higher business cost than a false alarm (False Positive), so optimizing the threshold maximizes true churn capture.
3. **Class Imbalance & Feature Engineering:**
   - **Balanced Weights:** Utilizing `class_weight='balanced'` in Logistic Regression dramatically improved minority class recovery, lifting recall from `0.567` up to `0.781`.
   - **Engineered Features:** Created customized service-count and contract-risk flags to improve model feature interactions.

---

### 🔹 Week 3: Model Optimization, Unsupervised Learning & Dimensionality Reduction
*(Note: Updated/refined during Week 4 to support seamless pipeline serialization and inference hand-offs).*

#### Key Insights
1. **Cross-Validation & Split Stability:**
   - Evaluating models across 20 random seeds revealed accuracy variations ranging from **79.2% to 83.1%**, proving single splits are unstable.
   - Switched to 5-fold cross-validation for robust evaluation:
     - **Logistic Regression (Tuned $C$):** $0.8464 \pm 0.0129$ AUC
     - **Random Forest (Random Search):** $0.8464 \pm 0.0114$ AUC
     - **XGBoost (Tuned):** $0.8502 \pm 0.0117$ AUC
2. **XGBoost Optimization:**
   - Randomized Search efficiently identified optimal tree parameters. Early stopping selected **247 trees**, preventing overfitting.
   - Final test AUC reached **0.8483**, falling comfortably within the expected 5-fold CV range ($[0.8268, 0.8736]$). Artifact saved as `churn_model.joblib`.
3. **Unsupervised Customer Segmentation ($k = 4$):**
   - **At-Risk New High-Spenders (43% churn):** *Action: Immediate onboarding & bundle discounts.*
   - **New Budget / Low-Engagement (32% churn):** *Action: Cross-sell add-ons at promo rates.*
   - **Loyal High-Value VIPs (14% churn):** *Action: Enroll in VIP/referral programs.*
   - **Stable Long-Term Budget (5% churn):** *Action: Maintain automated self-service.*
4. **Dimensionality Reduction (PCA):**
   - Exactly **15 out of 30 components** explain 90% of cumulative variance. PC1 loadings highlighted severe multicollinearity in "No internet service" dummy variables.

---

### 🔹 Week 4: Serving Pipeline & Web Deployment

#### Key Insights
1. **Zero Training-Serving Skew:**
   - Designed a dedicated serving pipeline with strict input schema validation and feature alignment.
   - Verified 100% feature parity with **0.000% discrepancy** across all 7,043 customer records between offline training batch predictions and online serving outputs.
2. **Interactive Decision Support (Churn Risk Advisor):**
   - Developed a custom **Streamlit** frontend allowing non-technical retention teams to perform single-customer risk scoring and batch uploads.
   - Built interactive **"What-If" scenario sliders** (contract changes, tech support add-ons, pricing shifts) to let agents simulate retention offers and watch the churn probability score adjust live.
3. **Cloud Deployment:**
   - Deployed live application directly from GitHub to **Streamlit Cloud**.

---

## 🛠️ Tech Stack & Tools
- **Programming Language:** Python 3.x
- **Machine Learning:** `scikit-learn`, `xgboost`, `pandas`, `numpy`, `imbalanced-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Deployment & Web UI:** `streamlit`, `joblib`
- **Environment:** Kaggle Notebooks & Streamlit Cloud
