# Credit Risk Prediction Model (Statlog German Credit Data)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.x-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 1. Project Overview

A machine learning solution for predicting credit risk using the German Credit Dataset. This project demonstrates the complete ML pipeline from exploratory data analysis to model deployment, with iterative improvements addressing class imbalance and business constraints.

---

## 2. The Business Problem & Data

The dataset consists of 1000 instances with 20 attributes (both numerical and categorical) describing loan applicants. The target variable classifies applicants into two categories:

* **Class 1 (Good Risk):** 700 instances (70%)
* **Class 2 (Bad Risk):** 300 instances (30%)

### Key Challenge: Class Imbalance & Cost

1.  **Imbalance:** The dataset is imbalanced (70/30). A naive model prioritizing accuracy would perform poorly, as it could simply guess "Good Risk" most of the time.
2.  **Business Cost:** The dataset's metadata specifies a **cost matrix** where misclassifying a "Bad Risk" customer as "Good Risk" is **5 times more costly** than misclassifying a "Good Risk" customer as "Bad."

Therefore, the primary objective is **not** to maximize overall accuracy, but to maximize the model's ability to identify the minority class ("Bad Risk")—a metric known as **Recall (for Class 0)**.

---

## 3. Methodology & Model Iteration

To solve this cost-sensitive problem, I iterated through several models to find the one that best balanced performance with the business objective.

### 3.1. Data Preparation

1.  **Target Mapping:** The target variable `class` was mapped from `{1: 1, 2: 0}`, where **1 = Good Risk** and **0 = Bad Risk**.
2.  **Feature Encoding:** The 13 categorical features were transformed using One-Hot Encoding (`pd.get_dummies`), expanding the feature space from 20 to 61 columns.
3.  **Data Split:** The data was split into 80% training and 20% testing sets using `stratify` to maintain the original class distribution in both sets.

### 3.2. Model Selection: The Journey to V3

* **Model V1 (Baseline Decision Tree):** A standard `DecisionTreeClassifier` achieved 68% accuracy but a **Recall of only 0.48 for Bad Risk**. It failed to identify more than half of the high-risk applicants.

* **Model V2 (Pruned Tree):** A tree with `max_depth=3` was trained to improve interpretability. While accuracy was 70%, the **Recall for Bad Risk dropped to 0.07**. This model was dangerously over-optimized for the majority class.

* **Model V3 (Final Model: Pruned + Balanced):** This model was designed to directly address the business problem.
    * `DecisionTreeClassifier(max_depth=5, class_weight='balanced')`
    * `max_depth=5`: Prunes the tree to prevent overfitting and improve generalization.
    * `class_weight='balanced'`: This key parameter automatically assigns a higher weight (penalty) to misclassifications of the minority class ("Bad Risk").

### 3.3. Final Model Evaluation (`modelo_risco_credito.pkl`)

The final model (V3) achieved a lower overall accuracy (66%) but was vastly superior for the business goal:

| Class | Precision | **Recall** | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Risco Ruim (0)** | 0.46 | **0.75** | 0.57 | 60 |
| Risco Bom (1) | 0.85 | 0.62 | 0.72 | 140 |
| **Accuracy** | | **0.66** | | 200 |

The **Recall for "Bad Risk" (0) jumped to 0.75**, meaning the model now correctly identifies 75% of high-risk applicants—a massive improvement over the baseline's 48%.

---

## 4. Key Insights & Interpretability

By visualizing the final Decision Tree (V3), we can extract clear, interpretable rules for credit approval.

<img width="1630" height="812" alt="image" src="https://github.com/user-attachments/assets/22333a76-11e4-4189-87b1-4a5322d92b6f" />

The model's logic highlights the most significant features found during the Exploratory Data Analysis (EDA):

1.  **`Attribute1` (Status da Conta Corrente):** This is the most critical feature. Applicants with no checking account (A14) or a low-balance account (A11) are immediately flagged as higher risk.
2.  **`Attribute3` (Histórico de Crédito):** Applicants with a "critical account" or previous delays (A34, A33) are strong predictors of default.
3.  **`Attribute2` (Duration in month):** Longer loan durations (e.g., > 33 months) are associated with higher risk.

---

## 5. How to Use This Project

You can either run the analysis from scratch using the notebook or use the pre-trained model (`.pkl`) to make predictions.

### 5.1. Requirements

First, install the necessary libraries:

```bash
pip install -r requirements.txt
```

### 5.2. Option A: Run the Full Analysis

Clone this repository:

```bash
git clone [https://github.com/](https://github.com/)[codeonthespectrum]/[predicao_risco_cc].git
cd [predicao_risco_cc]
```
Run the Jupyter Notebook:
```bash
jupyter notebook predicao_risco_cc.ipynb
```
This will execute the full EDA, data processing, and model training pipeline.

### 5.3. Option B: Load the Pre-trained Model
You can use the saved modelo_risco_credito.pkl file to make predictions on new data. The new data must be a Pandas DataFrame that has been one-hot encoded, resulting in 61 columns.

```bash
import joblib
model = joblib.load('modelo_risco_credito.pkl')
```
## 🎓 Learning Outcomes
This project demonstrates proficiency in:

- ✅ End-to-end ML pipeline: From raw data to production-ready model
- ✅ Handling class imbalance: Using class_weight and appropriate metrics
- ✅ Business-aware modeling: Optimizing for domain-specific cost functions
- ✅ Iterative development: V1 → V2 → V3 with clear justifications
- ✅ Model interpretability: Decision tree visualization and feature analysis
- ✅ Statistical analysis: EDA with correlation and distribution insights

## 📧 Contact
Gabrielly Gomes - [LinkedIn](https://www.linkedin.com/in/gabrielly-gomes-ml/) - gabrielly.gomes@ufpi.edu.br
