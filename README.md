# DS.v3.2.3.4

## Modelling Red Wine Quality with Regression

<img src=data/wine_quality.jpeg width=300 height=290>  
<br> 

This repository contains analysis of Red Wine Quality dataset. The purpose is to practice making an **explanatory** model using linear regression models.

#### Project overview
This project includes, first, exploratory data analysis of the Red Win Quality dataset. After splitting data to train (80%) and test (20%) subsets Multiple Linear Regression and Logistic Regression models were fitted. Five hypotheses were stated about possible effects of particular predictors. The interpretation of statistically significant coefficients is provided to understand each predictors' impact on perceived wine quality. Assumptions for each model were tested. Each model's performance on hold-out dataset was fulfilled, and its performance was evaluated.

#### Dataset
[Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) is taken from *Kaggle*.  

This dataset contains 1599 unique red wine samples with perceived wine quality estimate. Though there are 240 duplicated samples, but these are only wine samples with the same physicochemical features.   

The features in the dataset are all **numerical**: <br>
 * fixed acidity,
 * volatile acidity,
 * citric acid,
 * residual sugar,
 * chlorides,
 * free sulfur dioxide,
 * total sulfur dioxide,
 * density,
 * pH,
 * sulphates,
 * alcohol,
 * quality.

 #### Tools and libraries
 This project uses:
* **numpy**: For matrix manipulation.
* **pandas**: For data manipulation and cleaning.
* **matplotlib, seaborn**: For data visualization.
* **statsmodels, scipy, sklearn**: For regression modelling, statistical tests and metric calculation.

#### Key insights:
After performing both Multiple Linear (MLR), Ordered Logistic (OLR), and Binary Logistic Regression models we can state:

* There were identified the same set of statistically significant features that have an impact on perceived wine quality using both OLR and MLR models.
* MLR model failed for linearity, residuals' normality and homoskedasticity assumptions, causing our inference to be biased and not reliable.
* OLR model also failed in linearity of log odds and proportional odds assumptions, what also makes our inference not reliable.
* Despite our models' bias, we can say that the largest positive impact on perceived wine quality have alcohol and sulphates. However, volatile acidity has the largest negative impact on perceived wine quality.
* Final MLR model explains only about 36% of the total response variable variance.
* Accuracy of our final OLR model is ~57%. 
* Best OLR and MLR model performance is on most dominant 5th and 6th quality level classes because of huge data imbalance.
* To address data imbalance Binary Logistic regression model helped achieve balanced accuracy of 72% as compared to OLR model's 25%. 
* Pseudo R squared for OLR model is 19% as compared with 24% for Binary Logistic Regression model.

#### Suggestions for improvement:

* For explanatory purposes we sought for the simplest model for better interpretability, however, failing with assumptions for Ordinal and Multiple Linear Regression models make us conclude that further we should experiment including nonlinear terms of the predictors, search for meaningful interaction terms that could help us better capture nonlinearity in predictors-response relationship.

* Receiving more data for rarer wine quality samples could help as improve our final binary Logistic regression model and ensure model's robustness.
