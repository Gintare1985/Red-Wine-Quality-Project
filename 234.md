You have made it to the end of Module 2, well done! Now let’s put all of this knowledge into practice by solving a statistical inference problem. 

### Objectives
- Practice choosing variables and their transformations for a model.
- Practice fitting a linear regression model.
- Practice interpreting a fitted model’s coefficients and communicating uncertainty around them.

### Requirements
Your task is to analyze the [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) dataset and identify which variables influence perceived wine quality the most by fitting a statistical model. Keep in mind that this is not a predictive model (you are not interested in predicting wine quality as accurately as possible), but an explanatory model (you are interested in understanding factors that influence wine quality).

- Download the [dataset from Kaggle.](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- Describe the dataset. Understand and document clearly what each variable in the dataset measures. - Describe how and when the data was collated, what is the sample size, and any other important detail.
- Formulate at least one specific hypothesis (e.g. “fixed acidity of wine has a negative impact on perceived wine quality”). Also state the corresponding null hypothesis.
- Perform EDA. Inspect variables separately, plot how they correlate with the independent variable and one another. Check for multicollinearity.
- Separate a random 20% of your dataset as a hold-out.
Select variables and their transformations for your model. Provide arguments for why those variables were chosen and which transformations were applied.
- Fit the model using statsmodels. Print out the model summary and provide a written interpretation of key model fit statistics (such as R-squared) and coefficients. Explain which variables have the largest impact on the dependent variable, including their confidence intervals.
- Answer the hypothesis you stated in the beginning of the analysis.
- Inspect how the model performs on the hold out dataset.
- Provide suggestions for how your analysis could be improved.

### Evaluation Criteria
- Adherence to the requirements. How well did you meet the requirements?
- Visualization quality. Did you use charts effectively to support your recommendations? Are your visualizations properly labeled? Did you use colors effectively? Did you adhere to the principle of proportional ink?
- Modeling. Did you provide arguments for why some variables were included, excluded or transformed? 
- Interpretation of results. Are you able to explain where the uncertainty in your estimates comes from? Are you able to measure and communicate that uncertainty in the form of a confidence interval?
- Code quality. Was your code well-structured? Did you use the appropriate levels of abstraction? Did you remove commented out and unused code? Did you adhere to PEP8?
- Code performance. Did you use the suitable algorithms and data structures to solve the problems?

### Project Review
During your project review, you should present your project as if talking to a manager and senior data scientist working in your team. You will have to find the right balance between explaining your insights intuitively and diving deeper into the technical aspects of your work. 

During a project review, you may get asked questions that test your understanding of covered topics. Here are some examples:

- What is a confidence interval? Why do we need it? - Why is it not sufficient to just report the regression coefficients?
- What choices would you have made differently if you were solving a prediction problem?
- How did you choose which variables to include in your model and how to transform them?
- What is multicollinearity? Can you give an example? Why is it important to avoid multicollinearity in explanatory models?
- What is a confounding variable? Can you give an example?
- What is overfitting? How to detect and reduce overfitting?

For general project review guidelines, please refer to [this document.](https://turingcollege.atlassian.net/wiki/spaces/DLG/pages/537395951/Peer+expert+reviews+corrections)

### Additional Resources for this Sprint

- [ ] [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)
