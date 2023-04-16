# BH-PCMLAI-Module17

This repository contains the practical application assignment for Module 17.
Additional details are provided in this activity's Juniper Notebook.

Notebook link: https://github.com/ksolivenhub/BH-PCMLAI-Module11/blob/main/ksoliven%20-%20prompt_II%20-%20fin.ipynb


## Methodology

The activity is based from the CRISP-DM framework which follows the following high-level steps:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment
   
<center>
    <img src = images/crisp.png width = 30%/>
</center>


    
### Business Understanding
    
This step will allow us to create a goal based on the current needs of an individual/organization.
The goal for this activity is:

`To predict if the client will subscribe a term deposit or not`

The data used in this modeling activity is related with direct marketing campaigns of a Portuguese banking institution. 
Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed.
The marketing campaigns were based on phone calls.

Note: Term Deposits are deposits that are usually made for a few months to several years and reward you with guaranteed returns.
    
### Data Understanding

This step will allow us to have a general understanding of data by analyzing relationships, identify data quality issues, and create visualizations about the features found in the data set. Based on this activity, we would want to predict the sale price of cars and as such, this is a Regression problem.
    
<center>
    <img src = images/sklearn_algo_cs.png width = 70%/>
</center>

The following features were used an input in this study:

#### Input variables:

##### Bank client data:
1. age (numeric)
2. job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur",
                     "student","blue-collar","self-employed","retired","technician","services") 
3. marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
4. education (categorical: "unknown","secondary","primary","tertiary")
5. default: has credit in default? (binary: "yes","no")
6. balance: average yearly balance, in euros (numeric) 
7. housing: has housing loan? (binary: "yes","no")
8. loan: has personal loan? (binary: "yes","no")

##### Related with the last contact of the current campaign:
9. contact: contact communication type (categorical: "unknown","telephone","cellular") 
10. day: last contact day of the month (numeric)
11. month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
12. duration: last contact duration, in seconds (numeric)

##### Other attributes:
13. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
14. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
15. previous: number of contacts performed before this campaign and for this client (numeric)
16. poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

##### Output variable (desired target):
17. y - has the client subscribed a term deposit? (binary: "yes","no")

### Data Preparation

This step will ensure that the data has been preprocessed and that it has been prepared before performing any modeling activities.
For this project, I have performed the following preprocessing techniques:

1. Data Cleanup
    - The overall data set entries are `45211` records and feature count of '16'
       - There are no null values in the data set which is great
       - There is a feature named `poutcome` which I have removed as `~81%` of its records are `unknown` and as such, it will not provide useful insigts to the model and will only increase resource utilization if added 
    - The target data is imbalanced with `X% yes` and `Y% no`
       - During the data splitting for training and validation, I have normalized the distribution of target value to equally distribute the target between these datasets using `stratify`
       - This prompts that `Precision, Recall, and related metrics` are a better measure for performance in this activity
    - After the removal of 'poutcome', the total number of features is '15'
       - No additional number of features have been dropped from the dataset accordingly
    - Model-ready Data Preparation
       - Some features in the dataset are categorical in nature. As such, I have used `One Hot encoding/Target` functions to ensure that categorical variables are represented in binary nature.
         - It should be noted that One Hot encoding was done to binary features, including the target, and Target encoding was done on non-binary features.
       - After encoding, I have ensured that values are normalized by performing `scaling` and removed outliers by performing 'log transformation' on the datasets
         - This is an attempt to improve model performance
         - Note that the scaling was performed using the `StandardScaler()` function and `np.log(dataset+2)` where `2` is an arbitrary number to ensure no (or minimal) infinite /N/A values are generated within the data set
         - After this process was completed, a total of `45162` (dropped `49` after the log transformation) records are retained
 
### Modeling
<To Edit>

This step will focus on creating models to predict the target (term depositon subscription) based on testing data set.
For this activity, I have attempted to create multiple models and iterated them using HalvingRandomSearch (faster option) and GridSearchCV (slower option) to compare performance:

Notes:
1. I have used the same dictionary of parameters for both functions
2. Warnings are displayed to allow for troubleshooting (as required)
3. During the experimentation phase, some functions do produce `max_iter` warnings and as such, applied a higher number to it.
   - However, some outputs still do show the max_iter error and didn't have time to clean it up more due to time of model training execution and due to project time constraints
4. HalvingRandomSearch performed faster than GridSearchCV but wanted to compare performance for higher number of datasets

The training model used for this activity are as follows:

1. Logistic Regression
2. Stochastic Gradient Descent (SGD) Classifier
3. KNearest Neighbors
4. Decision Trees
5. Support Vector Machines (SVM)

#####################################    
    
<center>
    <img src = images/Attempt1.PNG width = 30%/>
</center>
    
In my first attempt, I have performed the following:
    
1. Perform a GridSearchCV over a Pipeline that uses PolynomialFeatures, another stage of SFS and Ridge Regression model.
    
2. I have performed two (2) test runs wherein each training model took between 1-4 hrs of training time.

Below is the param_dict for the GridSearchCV object:

```    
param_grid = {
    'poly__degree': [1, 2, 3],
    'sfs1__n_features_to_select': [3, 4],
    'model__alpha': [10, 0, 0.1, 0.01]
}
```    
The results are as follows:

```    
grid_search.best_params_
Best parameters: {'model__alpha': 100, 'poly__degree': 3, 'sfs1__n_features_to_select': 4}
    
grid_search.best_score_    
Best score: -0.9873754095247824

```    

When comparing error with target (test set):

```    
Test MAE: 0.5962240401620559 
```    
    
#### Attempt 2 (Jupyter Notebook not included)
---
   
<center>
    <img src = images/Attempt2.PNG width = 30%/>
</center>

In my second attempt, I have performed the following: 

    
1. Perform a GridSearchCV over a Pipeline that uses column(cat/num features) preprocessing techniques (One Hot Encoding and StandardScaler), 1st stage of SFS, PolynomialFeatures, and 2nd stage of SFS, and using the Ridge Regression model.    

2. I have performed two (2) test runs wherein each training model took between 1-4 hrs of training time.

Below is the param_dict for the GridSearchCV object:

    
Below is the param_dict for the GridSearchCV object:

```    
param_grid = {
    'poly__degree': [1, 2, 3],
    'sfs1__n_features_to_select': [3, 4],
    'model__alpha': [10, 0, 0.1, 0.01]
}
```    
The results are as follows:

```    
grid.best_params_
Best parameters: {'feature_selection2__n_features_to_select': 5, 'model__alpha': 10, 'model__random_state': 42, 'poly_expansion__degree': 2}
    
grid.best_score_    
Best score: -83226689944671.16
    
```    

When comparing error with target (test set):

 
    
    
    
    
### Evaluation
<To Edit>

For this section, the focus would be on my submission Attempt 1.   

The following are my findings that can potentially improve my model and other useful information that I can share with the client:
    
##### A. Internal Findings
1. The use of a powerful machine to run slightly complex models is a must to ensure processing speed
    - I have seen that my CPU and memory average at 100 and 90% respectively
2. We can revisit a better way to handle the following:
    - Maybe there is a more robust way to relate features that have thousands of unique entries to the data set
    - Maybe we can incorporate some items that I did with my **2nd attempt** with the **1st attempt** to further performance
        - Such as outlier trimming, pipeline instantiation, and others
3. Based on the final model, the most crucial features that determine the price are:
    - `condition`
    - `cylinders`
    - `drive`
4. I have selected an evaluation metric MAE to remove any bias related with outliers.
    - Mean Absolute Error (MAE) is a metric used to evaluate the performance of a regression model. It measures the average absolute difference between the predicted values and the true values. The lower the MAE, the better the model's performance, as it indicates that the model's predictions are closer to the true values.
    - `Best Model MAE: 0.596`
    
##### B. Client Information
1. I found that certain `price` entries is a bit ambiguous such that the cost is less than the normal value
    - Some values only between 1-3 digit values
2. The way the client record model is not the greatest
    - Unique models values is about 29k
    - Most of these are misspelled, a repetition of the manufacturer, NaN but has a manufacturer information, or incorrectly labeled
    - Some details from the manufacturer feature are missing but is found under model
3. I have seen that each state/region's car sales pricing is standardized
    - I am not sure if this is expected or not
4. VIN entries are duplicated
    - With this, some fields will be, in effect, not as useful
5. There is an interesting group that was not properly documented which are related to semi trucks
    - This could be strongly related to manufacturer, model, and odometer readings that have very high values


### Deployment
<To Edit>

Here is the summary report that can be provided to the client:
    
Goal: To determine the best parameters that will be used in creating an optimum model that will predict the appropriate value for used cars
    
Based on the results of this analysis, we can conclude that the model is able to predict the price of vehicles based on the following metrics with a minimal degree of error.
- `condition`
- `cylinders`
- `drive`

The following are the recommendations that can be used moving forward:
1. Use the metrics obtained from the model to perform car pricing initiatives within the organization
   - A simple formula can be obtained which are as follows:
  
  ```
  price_pred = 'condition'*(-0.05614311) +  'cylinders^2'*(0.04209613) + 'cylinders^3'*(0.01194092) + 'drive^3'*(0.04234781)
  ```  
    
   - Note that the variables `condition`, `cylinders`, and `drive` should be numerically represented which can be taken from the analysis
    
2. To improve future models, investment on data management must be considered
   - Price information has to be cleaned up as some values are only between 1-3 digit values
   - Cnsider improving data related to semi trucks to improve pricing predictions for this group accordingly
   - Ensure that the models feature needs to be standardized accordingly considering its 29k unique entries
