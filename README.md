# Credit Risk Classification

## Introduction
- The analysis aims to establish and assess a model that can identify and predict borrowers' creditworthiness.

## Machine Learning Model Information
1. Logistic Regression
2. XGBoost

## Model Evaluation Score
* Machine Learning Model 1 (Logistic Regression)
  * Balanced Accuracy: 0.944
  * Health Loan(label zero): precision(1.00),recall(1.00),f1-score(1.00)
  * High-risk Loan(label one): precision(0.87),recall(0.89),f1-score(0.88)

* Machine Learning Model 2 (Logistic Regression with Resampled Training Data)
  * Balanced Accuracy: 0.996
  * Health Loan(label zero): precision(1.00),recall(1.00),f1-score(1.00)
  * High-risk Loan(label one): precision(0.87),recall(1.00),f1-score(0.93)

* Machine Learning Model 3 (XGBoost Model with Original Imbalanced Dataset)
  * Test Accuracy: 0.995
  * Train Accuracy: 0.994
  * Health Loan(label zero): precision(1.00),recall(1.00),f1-score(1.00)
  * High-risk Loan(label one): precision(0.87),recall(1.00),f1-score(0.93)

