# Modeling Question and Answers

## 1)

### **What is Linear and Logistic Regression? How are they different?**

Linear regression is used to make a value prediction.

Logistic regression is used to make a binary classification. It is not predicting a value of *y*, but rather the probability that an outcome is a success or failure.



## 2)

### **Describe cross-validation and its role in model selection.**

A *model validation* technique for assessing how the results of a statistical analysis will generalize to an independent dataset. 

Estimating model prediction performance.

Mainly used when one wants to estimate how accurately a predictive model will perform in practice. 

In a single round, a model is given a partition of the dataset, called the 'training' data, which it uses to build or train its parameter estimates. The model is then run on the other partition of the dataset, the 'test' or 'validation' data, to see how well its prediction performs.

Sometimes referred to as *rotation estimation*, since you rotate through different training/test selections of the same dataset.

Multiple rounds of cross validation on different partitions reduces variablity, and the results can be combined and averaged to estimate a final model.

Can limit overfitting.

Used instead of conventional validation (a single 70/30 training/testing split) when there is not enough data to split without losing modeling or testing capacity. 

Cross-validation combines (averages) measures of fit (prediction error) to derive a more accurate estimate of model prediction performance.

**It can also be used to select modeling methods for the data:**

Cross-validation lets you compare different prediction models/methods in terms of their respective misclassifications/false-positives/error

This can help you detect models that are prone to over-fitting.


