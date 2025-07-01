# 🧬 Phenotype Prediction Using Tree-Based Models in the UK Biobank

## 🚀 Motivation

The ability to predict human phenotypes from genomic and environmental data has enormous potential for personalized medicine, risk stratification, and healthcare resource allocation. While genome-wide association studies (GWAS) have identified many relevant variants, integrating those with demographic and behavioral data in powerful machine learning models can significantly improve phenotype prediction. This project aims to benchmark tree-based models on the UK Biobank dataset to evaluate their effectiveness in phenotype prediction and interpretability.

## 📚 Overview

This repository contains code and analysis from our IEEE BIBM 2023 paper:  
**"A Tree-Based Approach to Phenotype Prediction Using the UK Biobank."**  
We explore ensemble-based methods like XGBoost and LightGBM to predict various clinical phenotypes and interpret the models using SHAP (SHapley Additive exPlanations) for feature contribution insights.

## 🧬 Data

All phenotypic data used in this study come from the **UK Biobank**, a large-scale biomedical database and research resource containing in-depth genetic and health information from over 500,000 UK participants aged 40–69. For phenotype selection and preprocessing, we used the publicly available [Stanford Biobank Engine](https://biobankengine.stanford.edu/), which allows for phenotype exploration and trait lookup.

Access to the UK Biobank data requires application and approval through [https://www.ukbiobank.ac.uk](https://www.ukbiobank.ac.uk).

## ⚙️ Methodology

We used tree-based machine learning models, including:

- XGBoost
- CatBoost
- LightGBM
- Decision Trees
- Random Forest

Each model was trained on genetic, demographic, and lifestyle variables. Evaluation was conducted using standard performance metrics such as AUC (for binary traits) and R² (for continuous traits). SHAP values were used for feature importance interpretation.

## 📈 Results

<div align="center">

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Phenotype</th>
      <th>AUC / R²</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>XGBoost</td>
      <td>Type 2 Diabetes</td>
      <td>0.81</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>Hypertension</td>
      <td>0.78</td>
    </tr>
    <tr>
      <td>RandomForest</td>
      <td>Smoking (ever vs never)</td>
      <td>0.76</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>Fasting Glucose</td>
      <td>0.23 (R²)</td>
    </tr>
  </tbody>
</table>

</div>

## 📝 Citation
If you use this work, please cite:

A. Meléndez et al., "Assessing Tree-Based Phenotype Prediction on the UK Biobank," 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Istanbul, Turkiye, 2023, pp. 3804-3810, doi: 10.1109/BIBM58861.2023.10385960.


## 🔗 Links

- [UK Biobank](https://www.ukbiobank.ac.uk/)
- [Stanford Biobank Engine](https://biobankengine.stanford.edu/)
