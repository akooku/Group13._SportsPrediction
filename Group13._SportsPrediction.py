# -*- coding: utf-8 -*-
"""Mid-Semester Project: Sports Prediction

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RzyHvKQZbdm9aKiQBKhofAE9h_sMlBfL

Firstly, we need to import all the important modules and functions needed for our project.
"""

# For data preprocessing and feature engineering
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# For model training
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# For evaluation
from sklearn.metrics import mean_absolute_error

# For saving the model
import pickle

"""Next, we will import the needed datasets for our project."""

# Import datasets
fifa_21 = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Intro to AI/players_21.csv")
fifa_22 = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Intro to AI/players_22.csv")

"""Now, we can start data preprocessing.

# <center>__Data Preprocessing__</center>
This entails:
*   Data Cleaning [removing useless variables]
*   Exploratory Data Analysis
*   Imputation
*   Encoding

Firstly, we will examine the dataset to have a general idea of how the features look like. This will help us know which columns to remove, impute, and encode.
"""

# Examining the dataset
fifa_21.head()

# Checking for more information about the dataset
fifa_21.info()

"""From the dataset, we see that there are 110 column entries and the dataset contains both numeric and non-numeric columns."""

# Checking all the column entries
fifa_21.info(verbose=True) # more detailed description of the dataset

"""The first thing to do after examining the dataset is to remove columns with more than 30% of their values missing."""

# Define the percentage threshold
percent = 30

# Calculate the percentage of missing values in each column
missing_percent_per_column = (fifa_21.isnull().sum() / len(fifa_21)) * 100

# Drop columns exceeding the threshold
columns_to_drop = missing_percent_per_column[missing_percent_per_column > percent].index
fifa_21.drop(columns=columns_to_drop, inplace=True)

# Examining the new dataframe
fifa_21.info(verbose=True)

"""Now we have 102 columns from 110, meaning 8 of the columns had more than 30% of their values missing.

We will split the remaining columns into numeric and non-numeric for imputation and encoding.
"""

# Splitting into numeric columns
numeric_columns = fifa_21.select_dtypes(include=['number'])
numeric_columns.info(verbose=True)

"""We notice that all the int types are filled whereas the float types contain missing values. We will impute the missing values with the mean of the non-missing values."""

# Imputing the missing values in the numeric dataframe
imp=SimpleImputer()
imp.fit(numeric_columns)
imputed_data=imp.fit_transform(numeric_columns)
numeric_columns=pd.DataFrame(imputed_data, columns=numeric_columns.columns)

# Examining the new dataframe
numeric_columns.info(verbose=True)

"""Now that we have imputed missing values and the numeric dataframe is complete, we can move on to the non-numeric columns."""

# Splitting into non-numeric columns
categorical_columns = fifa_21.select_dtypes(exclude=['number'])
categorical_columns.info(verbose=True)

"""We will fill in the missing values of this dataframe with the values of the non-missing values ahead of it using forward fill."""

# Imputing the missing object values
categorical_columns = categorical_columns.fillna(method='ffill', axis=0)
categorical_columns.info(verbose=True)

"""Now that the dataframe is complete, we can encode the non-numeric columns to convert categorical data into a numeric format that machine learning algorithms can process."""

# Encoding the object values to numeric data types
le = LabelEncoder()
for col in categorical_columns.columns:
    categorical_columns[col] = le.fit_transform(categorical_columns[col])
categorical_columns.head()

"""After imputing and encoding the numerical and non-numerical data types, we will combine them and check their correlation with the target variable using a linear regression model. This is where the feature engineering process starts."""

# Combining the two dataframes
combined_df = pd.concat([numeric_columns, categorical_columns], axis=1)
combined_df.head()

"""# <center>__Feature Engineering__</center>
This entails:
*   Correlation Analysis
*   Feature Importance
*   Feature Selection
*   Feature Scaling

Since we are using a RandomForest model to check the importance of the feature variables with the target variable, we will need to split the dataset.
"""

# Split target and feature variables
y = combined_df['overall']
X = combined_df.drop('overall',axis=1)

"""Then, we will scale these features to ensure that they have consistent scales and to prevent features with larger magnitudes from dominating the modeling process."""

# Scaling the data values for training
sc=StandardScaler()
scaled_data = sc.fit_transform(X)

# Transform it into a dataframe
X = pd.DataFrame(scaled_data, columns=X.columns)
X.head()

"""After scaling, we will train the model with these variables."""

# Train the model
model = RandomForestRegressor()
model.fit(X,y)

"""Next, we will retrieve the important features and sort them in descending order."""

# Retrieving important features
feature_names = X.columns
feature_importance = model.feature_importances_

# Sorting the feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Adjusting the dataframe to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Sorting the features in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df

"""Since most of the variables show low importance with the target variable, we will pick the top 5 variables, which have an importance greater than 0.01."""

# Select the top 5 features based on importance
top_5_features = feature_importance_df['Feature'].values[:6]
top_5_features

"""Now, we will replace X with just the feature variables."""

# Assign the top 5 features to X
X = X[top_5_features]

# Since age and dob are used to find a player's age, we will drop dob and use just age
X.drop('dob', axis=1, inplace=True)

X.head()

"""We are now ready to train the dataset.

# <center>__Training Models/Evaluation__</center>
This entails:
*   Data Splitting
*   Model Selection
*   Hyperparameter Tuning
*   Cross-Validation
*   Metrics - Mean Absolute Error (MAE)
"""

# Splitting the dataset for training and testing
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42)

"""**Model 1: Random Forest**"""

# Creating a RandomForestRegressor object
rf = RandomForestRegressor()
rf.fit(Xtrain,Ytrain)

# Evaluating the model
y_pred=rf.predict(Xtest)
mean_absolute_error(y_pred,Ytest)

"""**Model 2: XGBRegressor**"""

# Creating an XGBRegressor object
xgb = XGBRegressor()
xgb.fit(Xtrain,Ytrain)

# Evaluating the model
y_pred=xgb.predict(Xtest)
mean_absolute_error(y_pred,Ytest)

"""**Model 3: Gradient Boosting**"""

# Creating a Gradient Boosting Regressor object
gb = GradientBoostingRegressor()
gb.fit(Xtrain,Ytrain)

# Evaluating the model
y_pred=gb.predict(Xtest)
mean_absolute_error(y_pred,Ytest)

"""After evaluating the model, we see that the Random Forest performs best with an MAE value of **0.246**. Next, we will use the **GridSearchCV** to combine cross-validation with the grid search for hyper-parameter tuning and optimization of the three models.

**Model 1: Random Forest**
"""

# Defining the parameters
cv=KFold(n_splits=5)
PARAMETERS = {
    'n_estimators': [100,200],
    'max_depth': [10,20,30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True],
    'random_state': [0,45],
    'n_jobs': [-1]
}

# Defining and fitting the GridSearchCV object
gs_rf = GridSearchCV(rf, param_grid=PARAMETERS, cv=cv, scoring='neg_mean_squared_error')
gs_rf.fit(Xtrain,Ytrain)

# Checking the best model
best_regressor = gs_rf.best_estimator_
best_regressor

# Checking the best parameters
best_params = gs_rf.best_params_
best_params

# Evaluating the RandomForestRegressorModel using MAE
y_pred = gs_rf.predict(Xtest)
mean_absolute_error(y_pred,Ytest)

"""**Model 2: XGBRegressor**"""

# Defining the parameters
cv=KFold(n_splits=5)
PARAMETERS = {
    'n_estimators':[30,50,100,120],
    'max_depth':[2,5,10,12],
    'random_state':[0,2,15,45],
}

# Defining and fitting the GridSearchCV object
gs_xgb = GridSearchCV(xgb, param_grid=PARAMETERS, cv=cv, scoring='neg_mean_squared_error')
gs_xgb.fit(Xtrain,Ytrain)

# Checking the best model
best_regressor = gs_xgb.best_estimator_
best_regressor

# Checking the best parameters
best_params = gs_xgb.best_params_
best_params

# Evaluating the XGBRegressorModel using MAE
y_pred = gs_xgb.predict(Xtest)
mean_absolute_error(y_pred,Ytest)

"""**Model 3: Gradient Boosting**"""

cv=KFold(n_splits=5)
PARAMETERS = {
    'n_estimators':[30,50,100,120],
    'max_depth':[2,5,10,12],
    'random_state':[0,2,15,45],
}

# Defining and fitting the GridSearchCV object
gs_gb = GridSearchCV(gb, param_grid=PARAMETERS, cv=cv, scoring='neg_mean_squared_error')
gs_gb.fit(Xtrain,Ytrain)

# Checking the best model
best_regressor = gs_gb.best_estimator_
best_regressor

# Checking the best parameters
best_params = gs_gb.best_params_
best_params

# Evaluating the GradientBoostingRegressorModel using MAE
y_pred = gs_gb.predict(Xtest)
mean_absolute_error(y_pred,Ytest)

"""It seems the **Random Forest Model** still produces the best result, with an MAE score of **0.243**. Still, we will test the three fine tuned models on a completely new data set to see which one performs best.

# <center>__Test with new data set__</center>

First we need to make sure the new dataset contains only the columns used for training and testing the models.
"""

# Selecting the important columns from the fifa 22 dataset
used_columns = ['overall','value_eur','release_clause_eur','potential','movement_reactions','age']
fifa_22 = fifa_22[used_columns]
fifa_22.info()

"""From the 2022 FIFA dataset, we see that there are missing values in some of the columns. We need to impute those missing values. Since all the columns are numeric, we do not need to split the dataset."""

# Imputing the missing data values
imp=SimpleImputer()
imp.fit(fifa_22)
imputed_data=imp.fit_transform(fifa_22)
fifa_22=pd.DataFrame(imputed_data, columns=fifa_22.columns)

# Examining the cleaned dataset
fifa_22.info()

"""Next, we will split the dataset into training and testing."""

# Storing the target variable
Ytest_22 = fifa_22['overall']

# Drop the target variable from the database
fifa_22.drop('overall', axis=1, inplace=True)

"""Then we will scale the trained variables to ensure that they have consistent scales and to prevent features with larger magnitudes from dominating the modeling process."""

# Scaling the data values for training
sc=StandardScaler()
scaled_data = sc.fit_transform(fifa_22)

# Save the scaled_data
with open('scaled_data.pkl', 'wb') as file:
    pickle.dump(sc, file)

fifa_22 = pd.DataFrame(scaled_data, columns=fifa_22.columns)

# Assign the feature variables to X
Xtest_22 = fifa_22

"""Testing with the fine tuned models."""

# Using the RandomForestRegressor model to test the 2022 dataset
y_pred = gs_rf.predict(Xtest_22)
mean_absolute_error(y_pred,Ytest_22)

# Using the XGBRegressor model to test the 2022 dataset
y_pred = gs_xgb.predict(Xtest_22)
mean_absolute_error(y_pred,Ytest_22)

# Using the GradientBoostingRegressor model to test the 2022 dataset
y_pred = gs_gb.predict(Xtest_22)
mean_absolute_error(y_pred,Ytest_22)

"""The **Random Forest Model** performs best with the new dataframe, giving an MAE of **0.567**, so we will save this model and use it for deployment."""

# Saving the RandomForestRegressorModel
filename = 'rf_model.pkl'
pickle.dump(gs_rf, open(filename, 'wb'))

"""# <center>__Deployment__</center>

The deployment part of the code will be ran on a different file.
"""

# Saving the ytest and ypred to load in our python file
combined_values = {'Ytest':Ytest, 'y_pred' : y_pred}
values_df = pd.DataFrame(combined_values)

# Changing the dataframe to a csv file to load
values_df.to_csv('Ytest_and_y_pred.csv',index = False)