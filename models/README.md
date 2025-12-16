# Machine Learning Models

This directory contains machine learning models trained on extracted EEG features
for task load / weight analysis.

## Classification
- **svm_loso_classification.py**
  - SVM with RBF kernel
  - PCA-reduced EEG bandpower features
  - Leave-One-Subject-Out (LOSO) validation
  - Balanced accuracy and confusion matrix evaluation

## Regression
- **mlr_loso_regression.m**
  - Ridge-regularized Multiple Linear Regression
  - Lagged EEG features
  - K-fold cross-validation for lambda selection
  - Leave-One-Subject-Out (LOSO) evaluation

