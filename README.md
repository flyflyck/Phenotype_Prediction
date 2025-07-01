# 🧬 Phenotype Prediction Using Tree-Based Models in the UK Biobank

This repository accompanies the paper:

**"A Tree-Based Approach to Phenotype Prediction Using the UK Biobank"**  
📍 *Presented at IEEE BIBM 2024*  
👤 *Author: Alex Melendez (Stanford University)*

---

## 📌 Overview

This project investigates the use of tree-based machine learning models (e.g., XGBoost, LightGBM) for predicting complex phenotypes from genetic, demographic, and behavioral data using the UK Biobank. We emphasize both predictive performance and interpretability using SHAP values to identify impactful features and patient subgroups.

---

## 🎯 Objectives

- Predict binary, categorical, and continuous phenotypes using ensemble tree models.
- Evaluate model performance using cross-validation and test sets.
- Interpret predictions using SHAP values to uncover key features and stratify patient subgroups.

---

## ⚙️ Methods

We trained tree-based models on selected SNPs and covariates. Each model was evaluated using 5-fold cross-validation and tested on held-out data. SHAP values provided local and global model explanations.

Target phenotypes included:

- **Type 2 Diabetes (binary)**
- **Fasting Glucose (continuous)**
- **Smoking Status (categorical)**

---

## 📈 Results Summary

| Model        | Phenotype             | AUC / R² |
|--------------|------------------------|----------|
| XGBoost      | Type 2 Diabetes        | 0.81     |
| LightGBM     | Hypertension           | 0.78     |
| RandomForest | Smoking (ever vs never)| 0.76     |
| XGBoost      | Fasting Glucose        | 0.23 (R²)|

Top predictors included BMI, age, and known disease-associated SNPs. SHAP plots revealed clinically meaningful patterns consistent with epidemiological literature.

---
