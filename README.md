# DS.v3.2.3.4

## Modelling Red Wine Quality with Regression

<img src=data/wine_quality.jpeg width=300 height=290>  
<br> 

This repository contains analysis of Red Wine Quality dataset. The purpose is to practice making an **explanatory** model using Multiple Linear Regression and Ordered Logistic Regression models.

#### Project overview
This project includes first exploratory data analysis of the Red Win Quality dataset. After splitting data to train (80%) and test (20%) subsets Multiple Linear Regression and Logistic Ordered Regression models were fitted. Five hypotheses were stated about possible effect of particular predictors. The interpretation of statistically significant coefficients is provided to understand each predictors' impact on perceived wine quality. Assumptions for each model were tested. Model's performance on hold-out dataset was fulfilled and model's performance was evaluated.

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
After performing both Multiple Linear and Ordered Logistic Regression models we can state:

* There were identified the same set of statistically significant features that have an impact on perceived wine quality using both Logistic and Multiple Linear Regression Models.

* Multiple Linear Regression model failed for linearity, residuals' normality and homoskedasticity assumptions, causing our inference to be biased not reliable.

* Ordered Logistic Regression Model also failed in linearity of log odds and proportional odds assumptions, what also makes our inference not reliable.

* Despite our models' bias, we can say that the largest positive impact on perceived wine quality have sulphates and alcohol. However, volatile acidity has the largest negative impact on perceived wine quality.

* Final Multiple Linear Regression model explains only about 36% of the total response variable variance.

* Accuracy of our final Ordered Logistic Regression model is ~57%. Pseudo R squared is 19%, suggesting a moderate improvement in the log-likelihood over only-intercept model.

* Best OLR and MLR model performance is on most dominant 5th and 6th quality level classes because of high data imbalance.

* OLR model on the test data correctly identified 49% of the cases, ~40% were misclassified to the nearest lower or higher perceived quality class.

* MLR model on the test data correctly identified 59% of the cases, ~40% were misclassified to the nearest lower or higher perceived quality class.

* OLR model misclassified sample to the second nearest lower or higher perceived quality class for 9% samples, multiple linear regression - for 2.5% samples.

#### Suggestions for improvement:

For explanatory purposes we seek for the simplest model for better interpretability, but failed assumptions for Ordinal and Multiple Linear Regression models make us conclude that more complex model or just including polynomial, meaningful interaction or spline terms could help us capture nonlinearity in predictors response relationship.
