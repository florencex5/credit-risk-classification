# Credit Analysis Report


## Introduction & Data Preprocessing
- The analysis aims to establish and assess a model that can identify and predict borrowers' creditworthiness. 
- Firstly, separating the target variable(`loan status`) from the entire set of features. The model is constructed using historical lending data with features (e.g. `loan size`,`interest rate`,`borrower income`, `debt to income ratio`, `number of accounts`,`derogatory marks` and `total debt`), with an initial choice of a logistic regression machine learning model.
- Recognizing the data's inherent imbalance for target variable (`loan status`), which the original data set only has 3.33%(2500/75036) loans showing high risk. And LogisticsRegression Machine Learning model is sensitive to class imbalance. A random oversampling technique is employed to rectify this issue by generating new samples in the classes which are under-represented. 
-  Additionally, an optional XGBoost model is introduced as an alternative approach to address the class imbalance issue.


## Model Training
- When splitting the data for training and testing, the default test data size of 0.25 was used. This means that 25% of the dataset was reserved for testing, while the remaining 75% was used for model training and evaluation.


## Model Evaluation

* Machine Learning Model 1 (Logistic Regression)
  * Balanced Accuracy: 0.944
  * Health Loan(label zero): precision(1.00),recall(1.00),f1-score(1.00)
  * High-risk Loan(label one): precision(0.87),recall(0.89),f1-score(0.88)
  
- The logistic regression model has around 94.4% balanced accuracy score.
- According to classification report, when classifying healthy loans ("label zero"), the model achieved perfect precision, recall, and F1-score, all equal to one. However, when dealing with high-risk loans ("label one"), the precision score drops down to 0.87. This indicates that the model only correctly identified 87% of loans labeled as high risk. The recall score of 0.89 ("label one"), implying its proficiency in capturing all instances of loans with high risk levels. As a result, the F1-score of high-risk loans("label one") only stops at 0.88, which leaves some space for minimizing the chances of erroneously categorizing healthy loans as risky.


* Machine Learning Model 2 (Logistic Regression with Resampled Training Data)
  * Balanced Accuracy: 0.996
  * Health Loan(label zero): precision(1.00),recall(1.00),f1-score(1.00)
  * High-risk Loan(label one): precision(0.87),recall(1.00),f1-score(0.93)

- Upon the implementation of random oversampling, the logistic regression model reached around 99.6% balanced accuracy score.
- When classifying healthy loans ("label zero"), the model continued achieved perfect precision, recall, and F1-score, all remaining at one. Conversely, in the context of high-risk loans (label one), the precision score remained at 0.87. Notably, the recall score improved to a perfect 1.0, indicating the model's ability to capture all high-risk loans accurately. The F1-score for high-risk loans stands at 0.93, demonstrating the model's capacity to maintain a great performance after addressing imbalanced class (i.e. high-risk loans) through random oversampling. 
- Furthermore, in light of concerns regarding overfitting associated with random oversampling, cross-validation was employed, revealing that all five folds report an F1-score of approximately 99%, indicating effective mitigation of overfitting issues.
  

## Recommendation

* Machine Learning Model 3 (XGBoost Model with Original Imbalanced Dataset)
  * Test Accuracy: 0.995
  * Train Accuracy: 0.994
  * Health Loan(label zero): precision(1.00),recall(1.00),f1-score(1.00)
  * High-risk Loan(label one): precision(0.87),recall(1.00),f1-score(0.93)
  
- I would recommend the adoption of the XGBoost model. While the performance difference between the logistic regression and XGBoost is marginal, XGBoost is tailored to excel in scenarios with imbalanced data distributions. By assigning higher weights to the minority class (e.g. high-risky loans) with its hyperparameter "scale_pos_weight", XGBoost actively prioritizes the accurate classification of these critical cases. Unlike the logistic regression model, which improved its performance only after applying random oversampling.
- In addition, XGBoost's also offers insights into the data's interpretability, such as feature importance analysis. Based on the result, the top three influential features are interest rate (0.990), loan size (0.006), and borrower income (0.004), with interest rate's imporatance score(0.990), higher interest rates could lead to loans being classified differently,as indicated by the XGBoost model.
