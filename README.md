# 💳 Credit Risk Decision Tool

> End-to-end data science consulting project — built to demonstrate McKinsey/BCG-style analytical thinking applied to a real financial services problem.

## 📌 Problem
UK lenders lose billions annually to credit defaults. Manual scoring is slow, inconsistent, and fails to quantify decision uncertainty. This tool provides an automated, data-driven decision support system for lending approval.

## 🔍 Approach
- **Dataset:** 30,000 real credit card customers (UCI Machine Learning Repository)
- **Models trained:** Logistic Regression, Random Forest, Gradient Boosting
- **Key technique:** Cost-optimised threshold selection rather than raw accuracy maximisation
- **Class imbalance:** Handled via SMOTE oversampling
- **Feature engineering:** Utilisation ratio, payment trend, missed payments count

## 📊 Results
| Metric | Value |
|--------|-------|
| Best model | Gradient Boosting |
| AUC-ROC | 0.771 |
| Optimal threshold | 0.25 |
| Default detection rate | 89.4% |
| Cost reduction vs baseline | 61% |
| Estimated saving per 6,000 applications | £12,068,000 |

## 🎯 Recommendation
Deploy model with a three-tier decision system:
- ✅ **Auto-approve:** risk score < 0.25
- ⚠️ **Human review:** risk score 0.25–0.45
- ❌ **Auto-decline:** risk score > 0.45

## 🚀 Live Demo
[Click here to try the interactive app](#) ← we'll add this link after Streamlit deployment

## 🛠️ Tech Stack
Python · scikit-learn · Gradient Boosting · SMOTE · Streamlit · pandas · seaborn

## 📁 Structure
