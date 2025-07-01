# 🧬 Phenotype Prediction Using Tree-Based Models in the UK Biobank

## 🚀 Motivation

Predicting human phenotypes from genomic and environmental data holds significant promise for personalized medicine, risk stratification, and efficient allocation of healthcare resources. Although genome‐wide association studies (GWAS) have identified numerous relevant genetic variants, integrating these findings with demographic and behavioral data through advanced machine learning approaches can substantially improve prediction accuracy. This project benchmarks a variety of tree-based algorithms on the UK Biobank dataset to assess their predictive performance and interpretability.

## 📚 Overview
This repository accompanies the IEEE BIBM 2023 paper titled “Assessing Tree-Based Phenotype Prediction on the UK Biobank.” We evaluate a suite of ensemble and boosting methods—including XGBoost, LightGBM, CatBoost, AdaBoost, Random Forest, and Extra Trees—alongside decision trees and linear models. Performance is measured using standard metrics (AUC for binary traits, R² for continuous traits), and model interpretability is achieved via SHAP (SHapley Additive exPlanations) to elucidate feature contributions.

## 🧬 Data

All genetic and phenotypic information originates from the UK Biobank, a comprehensive biomedical resource comprising over 500,000 participants aged 40–69 years. We applied for and received access through the UK Biobank application portal. Phenotype selection and data preprocessing were facilitated by the Stanford Biobank Engine, which provides tools for exploring and filtering traits of interest.

## ⚙️ Methodology

Our predictive framework utilizes nine tree-based algorithms, ranging from simple decision trees to sophisticated gradient-boosting machines. Each model is trained on combined sets of genetic variants, demographic covariates (such as age and sex), and lifestyle factors. A rigorous hyperparameter tuning process, based on multi‐objective optimization and five-fold cross-validation, ensures that each algorithm operates under its optimal parameter configuration. To explore the balance between prediction accuracy and computational efficiency, we employ Random Feature Selection (RFS), varying the number of genetic variants included and observing its impact on both model performance and runtime.

## 📈 Results

We report results for four representative binary and continuous phenotypes to illustrate key findings. Without hyperparameter tuning, gradient-boosting methods such as LightGBM and HGB outperform linear approaches, while Random Forest excels among tree ensembles for binary outcomes. Hyperparameter optimization further amplifies the performance gap, enabling tree-based models to surpass the performance of sparse linear methods like SNPnet, particularly for continuous traits. Incorporating age and sex as covariates yields additional gains in predictive accuracy, with CatBoost, LightGBM, and HGB emerging as the top performers overall.

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
      <td>Decision Trees</td>
      <td>Fasting Glucose</td>
      <td>0.23 (R²)</td>
    </tr>
  </tbody>
</table>
</div>

## 📝 Citation
If you use this work, please cite:
Meléndez A, López C, Bonet D, Sant G, Marquès F, Rivas M, Mas Montserrat D, Abante J, Ioannidis AG. Assessing Tree-Based Phenotype Prediction on the UK Biobank. In: 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Istanbul, Turkey; 2023. p. 3804–3810. [doi](10.1109/BIBM58861.2023.10385960).


## 🔗 Relevant Links
Access the UK Biobank at https://www.ukbiobank.ac.uk and explore phenotypes via the Stanford Biobank Engine at https://biobankengine.stanford.edu.
