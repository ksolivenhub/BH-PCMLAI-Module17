# BH-PCMLAI-Module17

This repository contains the practical application assignment for Module 17.
Additional details are provided in this activity's Juniper Notebook.

Notebook link: https://github.com/ksolivenhub/BH-PCMLAI-Module17/blob/main/ksoliven-practical-application-v1.ipynb
<br>
Note: Not being displayed correctly in GitHub as I have added warnings during model training to perform troubleshooting which increased the size and GitHub is unable to display accordingly

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

**The classification goal is to predict if the client will subscribe a term deposit (variable y).**
<br>
**For this project, it would be great to have a score of `80%` accordingly**

The data used in this modeling activity is related with direct marketing campaigns of a Portuguese banking institution. 
Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed.
The marketing campaigns were based on phone calls.

Note: Term Deposits are deposits that are usually made for a few months to several years and reward you with guaranteed returns.
    
### Data Understanding

This step will allow us to have a general understanding of data by analyzing relationships, identify data quality issues, and create visualizations about the features found in the data set. Based on this activity, we would want to predict the classification of term deposit subscriptions and as such, this is a Classification problem.
    
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
    - The target data is imbalanced (see image below)
       - During the data splitting for training and validation, I have normalized the distribution of target value to equally distribute the target between these datasets using `stratify`
       - This prompts that `Precision, Recall, and related metrics` are a better measure for performance in this activity
    - After the removal of 'poutcome', the total number of features is '15'
       - No additional number of features have been dropped from the dataset accordingly
    - Model-ready Data Preparation
       - Some features in the dataset are categorical in nature. As such, I have used `One Hot encoding/Target` functions to ensure that categorical variables are represented in binary nature.
         - It should be noted that One Hot encoding was done to binary features, including the target, and Target encoding was done on non-binary features.
       - After encoding, I have ensured that values are normalized by performing `scaling` and removed outliers by performing 'log transformation' on the datasets
         - This is an attempt to improve model performance
         - Note that the scaling was performed using the `StandardScaler()` function and `np.log(dataset+2)` where `2` is an arbitrary number to ensure no (or minimal) infinite / N/A values are generated within the data set
         - After this process was completed, a total of `45162` (dropped `49` after the log transformation) records are retained

<center>
    <img src = images/dataimbalance.png width = 50%/>
</center>
<br>

### Modeling

This step will focus on creating models to predict the target (term depositon subscription).
For this activity, I have attempted to create multiple models and iterated them using HalvingRandomSearch (faster option) and GridSearchCV (slower option) to compare performance:

Notes:
1. I have used the same dictionary of parameters for both functions
2. Warnings are displayed to allow for troubleshooting (as required)
   - This have affected the GitHub view of the notebook
   - Please download the Jupyter Notebook and view separately
3. During the experimentation phase, some functions do produce `max_iter` warnings and as such, applied a higher number to it.
   - However, some outputs still do show the max_iter error and didn't have time to clean it up more due to time of model training execution and due to project time constraints
4. HalvingRandomSearchCV performed faster than GridSearchCV but wanted to compare performance for higher number of datasets

The training model used for this activity are as follows:


1. Logistic Regression
2. Stochastic Gradient Descent (SGD) Classifier
3. KNearest Neighbors
4. Decision Trees
5. Support Vector Machines (SVM)

Note: No additional functions were added within the Pipeline when optimizing these models.
    
#### Modeling Results

Below are the parameter dictionary for both the HalvingRandomSearchCV and GridSearchCV objects:


| Logistic Regression | SGD Classifier | KNN | Decision Trees | SVM |
| --- | --- | --- | --- | --- |
| {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'multi_class': ['auto', 'ovr', 'multinomial'], 'max_iter': [100, 1000, 10000], 'fit_intercept': [True, False]} | {'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1], 'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']} | {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [10, 20, 30, 40, 50]} | {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 20, 50], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1, 2, 4], 'max_leaf_nodes': [5, 10, 20]} | {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

##### HalvingRandomSearchCV Run

In my first run, I have used HalvingRandomSearchCV to iterate the pre-defined parameter dictionaries for the five (5) models.

The results are as follows:

<center>
    <img src = images/hrd_1_results.png width = 80%/>
</center>

    
##### GridSearchCV Run

In my second run, I have used GridSearchCV to iterate the pre-defined parameter dictionaries for the five (5) models.

The results are as follows:

<center>
    <img src = images/gcv_1_results.png width = 80%/>
</center>

##### Initial Notes:

1. We are looking at the highest value of `Recall` - which can be seen using a `DecisionTree` model
2. The values from both optimization functions did not have much significant difference
3. HalvingRandomSearchCV was able to get the optimal values in a relatively fast rate compared with GridSearchCV
4. I have not disabled the warnings in this section to review and significant errors and rectify them accordingly
   - This have affected the GitHub view of the notebook
   - Please download the Jupyter Notebook and view separately      
5. During HalvingRandomSearchCV fitting, it produced several errors related to scoring
    - Note 1: This was a result mostly of using the scoring metric as `roc-auc`
    - Note 2: I have displayed all metrics as a way to compare values between trained models
6. During GridSearchCV fitting, it produced several errors related to fit warnings and max_iter
    - Note 1: This could be somehow related to the rate/way at which the model converges  
    - Note 2: I have displayed all metrics as a way to compare values between trained models
7. With SVC optimized by HalvingRandomSearchCV, `ROC AUC was valued NaN`
8. With SVC optimized by GridSearchCV, `Precision, Recall, and F1 were valued 0`
9. The goal is to make the project as simple as possible while yielding to high `Recall` score related to the Business Goal   
    
**Based on these factors, we can go ahead and use/optimize the `Decision Tree model`**

##### Decision Tree Model Optimization:

At this time, the Recall score of our Decision Tree model is at `42.91%`. I have tried to optimize it using the following techniques:
1. Resampling Techniques
   - This was done via `RandomOverSampler` and `RandomUnderSampler`
   - Improved the performance of the model by `~40%`
2. Feature Selection
   - This was done via `L1 Regularization`
   - Improved the performance of the model by reducing complexity of hyperparameters
3. Experimentation
   - I have tried different modifications to the Decision Tree model and have yielded to what I consider the `best model`
      - In terms of `Model Performance and Feature Complexity`

<center>
    <img src = images/best_model1.PNG width = 80%/>
</center>  
    
### Evaluation

The following are my findings that can potentially improve my model and other useful information that I can share with the client:
    
##### A. Internal Findings
1. The use of a powerful machine to run slightly complex models is a must to ensure processing speed
    - This is specially true for `SVC, KNN, and SGDClassifier` (arranged from highest execution time to lowest)
    - Experimentation can be done with these models but a significant blocker is the compute resource and time of execution
2. Based on the final model, the most important features that allows for predicting the target are as follows:
    - `job`
    - `housing`
    - `contact`
    - `month`
    - `duration`
3. `DecisionTree` was the best model - however, the model has to be optimized to improve the `Recall` metric
4. Applying `resampling techniques` for this problem have significantly improve the model prediction by almost `~40%` from the base Recall score
5. Applying `feature selection` allowed for the complexity of parameters yielding a higher recall score to be reduce, improving overall performance of the model
    - The use of `feature selection` with classification problems will not affect the results of the model even if used early on the process and could even improve performance
6. Using `GridSearchCV` and `HalvingRandomSearchCV` did not produce significant different on Recall scores
    - HalvingRandomSearchCV was able to get the optimal values in a relatively fast rate compared with GridSearchCV
7. Selecting the best score is not enough as we have to ensure that the best parameters would make sense
    - I have extracted the `cv_results`, analyzed which set of parameters does makes sense from the top ranked scores, and have selected it as the best parameters for my final model
8. The final model yielded an `87.34%` recall score, which matches with the goal for this project
9. We can revisit a more robust way to handle the following:
    - Sampling - as resampling techniques used namely `ROS/RUS` either trims the majority target data or creates random duplicates of the minority target data - which can create less optimal solutions
    - Normalization - as the normalization technique that I used namely `log-transformation` trimmed some data (but this has been minimized with trial/error)
    - Outlier Trimming - instead of the `log-transform` used, a consideration of just removing the outliers directly and testing could potentially yield to interesting results
    - Model Selection - this is affected with the compute resource that I use when training the model
10. Some information have minority records that are unknown (that was retained), which could have been trimmed and tested if it will improve performance

##### B. Client Information
1. Majority of the dataset is `imbalanced`
    - As this is a production data (of bank clients related to term deposit subscription), there is dataset limitation with respect to this factor
2. Some features have `unknown` record values
    - This could be improved by the bank in order to ensure that a clean data is available for data analysis
3. `May` is a busy month for attempting to convince clients for a term deposit subscription
    - This is an assertion based on this one-year data snapshot and can be used to improve activity and resource allocation when trying to contact clients to convince for a term deposit subscription

### Deployment

Here is the summary report that can be provided to the client:
    
Goal: **The classification goal is to predict if the client will subscribe a term deposit (variable y).**

    
Based on the results of this analysis, we can conclude that an `optimized DecisionTree model` is able to predict the classification of the client (subscription to term deposit) with a `Recall` score fo `87.34%` based on the following factors:
   - `job`
   - `housing`
   - `contact`
   - `month`
   - `duration`
    
To improve future models, investment on data management and model resource utilization must be considered
   - Unknown values have to be cleaned up as this will not offer additional insight in this context
   - Ensure that sufficient compute resources are invested to explore other alternatives if a higher degree of Recall score is considered
   - To create a more robust model, explore options to allow the improvement of Precision and/or F1 score as required.
