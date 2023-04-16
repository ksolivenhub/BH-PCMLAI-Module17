# BH-PCMLAI-Module11

This repository contains the practical application assignment for Module 11.
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

`To determine the best parameters that will be used in creating an optimum model that will predict the appropriate value for used cars`
    
### Data Understanding

This step will allow us to have a general understanding of data by analyzing relationships, identify data quality issues, and create visualizations about the features found in the data set. Based on this activity, we would want to predict the sale price of cars and as such, this is a Regression problem.
    
<center>
    <img src = images/sklearn_algo_cs.png width = 70%/>
</center>

### Data Preparation
    
This step will ensure that the data has been preprocessed and that it has been prepared before performing any modeling activities. For this project, I have used the following preprocessing techniques. Note that I have tried two (2) attempts

- **Attempt 1 (My Submission)**
- **Attempt 2**
    
---    
    
1. General data cleaning
    - The overall data set entries are `426880` records
       - There is a feature named `VIN` - where VIN is a unique ID that identifies a vehicle
       - The removal of VIN has also allowed further insights that some columns are not relevant with car price
    - Some features that are irrelevant to the car price analysis have been removed accordingly
       - This has been brought by insights during data review, such as that of VIN, features that are not meaningful to
         the analysis, features that have excessive NaN values (e.g., `size`), features that have all NaN values except for the `price`, and features that have excessive amount of categories (e.g., `model`)
       - The data mentioned removed was a calculated risk to improve overall performance of the model
    - Something that I have done unique on my attempts are as follows:
        - **Attempt 1**
            - Removing the `model` feature represents a significant loss in data. Other than the fact that the number of categories are excessive, there are also mispelled entries which adds up to the complexity of cleaning it. I have attempted to clean this up using `manual checks and SpellCheckers` but to no avail.
            - The results of my cleaning have allowed me to reduce `model` feature entries with NaN values from 29k to 24k and have added new entries on the manufacturer
                These new entries mostly represent `Semi Trucks\Cargo Trucks` which is a good insight 
        - **Attempt 2**
            - I have removed the `year` feature and have computed and added `age` into the dataframe in an attempt to improve performance

    
       
2. Preprocessing (Imputation/Encoding/Normalization/Scaling)
    - Most features in this data set are categorical in nature. As such, I have used the `One Hot encoding/Target` functions to ensure that categorical variables are represented in binary nature.
    - Numerical features appears to be skewed and in an attempt to normalize values, I have performed `log` transform to the features
    - Many features have values that are NaN. These NaN represents around ~20% (or below) of each feature records in this data set. The use of imputation using the mode/Iterative Imputation will allow automatically populating of these NaN values to allow a more meaning data analysis.
        - For the mode imputation, this may introduce a bias in the data modeling activities
    - To prepare the data set for modeling, I have performed `scaling` to the data set to improve overall performance and reduce variance to all features (i.e., setting reference to a mean = 0)
    - Something that I have done unique on my attempts are as follows:
        - **Attempt 1**
            - For categorical features that have less than 5k NaN values, I have filled it with the mode
            - For categorical features that have more than 5k NaN values, I have filled used Iterative Imputation
                - In order to do Iterative Imputation, all categorical feature records must be coverted to a numerical value where I have used Target Encoding
            - I have performed Log Transform and Scaling outside of the pipeline for all features including `price` to ensure that everything is normally distributed  
        - **Attempt 2**
            - I have removed outliers from the `Odometer` feature. The log transform has been taken and outliers have been trimmed in an attempt to improve performance.
            - I have filled all NaN values from the categorical features to its respective mode.
            - I have used One Hot encoding and have instantiated it within the pipeline

    
3. Feature Selection (Initial)
    - I have used a variety of techniques as an experiment and are as follows:
        - **Attempt 1**
            - VIF
                - The results showed values less than 5, which means that multicolinearity does not exist
            - SFS
                - The results showed that `condition, cylinders, title_status, transmission, and drive` are features of interest
            - Lasso feature selection
                - The results are similar with SFS
        - **Attempt 2**
            - I have integrated Encoding (One Hot), Scaling, SFS (initial), PolynomialFeatures, and SFS (main) within the Pipeline

### Modeling
    
This step will focus on creating models to predict the car price based on testing data set.
For this activity, I have attempted to create multiple models based on my experimentation.

Regression function used: 
1. Ridge

Pipeline Steps used: 
1. Polynomial feature expansion
2. Sequential feature selection
3. Lasso feature selection
4. Model Regression
    
To automate the process, a GridSearchCV object is used to iterate over a predefinied dictionary of parameters to acquire the best model. Based on the GridSearchCV training:
    
#### Attempt 1 (My Submission)
---
    
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
