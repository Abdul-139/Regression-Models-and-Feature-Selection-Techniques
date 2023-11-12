# -*- coding: utf-8 -*-
"""Decision Tree Regression Template

This script performs Decision Tree Regression on a given dataset after cleaning and feature selection.

# 1. Load a dataset
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset here
# Example: data = pd.read_csv('your_dataset.csv')

"""# 2. Clean the data"""

# Check for missing values
data.isnull().sum()

# Drop missing values and irrelevant columns
numerical_data = data.drop(['cmpdname', 'isosmiles', 'mw', 'exactmass', 'monoisotopicmass'], axis=1)
data_no_mv = numerical_data.dropna(axis=0)

# Display descriptive statistics
data_no_mv.describe(include='all')

# Display information about the cleaned data
data_no_mv.info()

# Display columns of the cleaned data
data_no_mv.columns

# Selected columns for analysis
cols = ['xlogp', 'polararea', 'heavycnt', 'hbonddonor', 'hbondacc', 'rotbonds']

# Visualize pairplots for selected columns
sns.pairplot(data_no_mv[cols])

# Calculate correlation matrix
data_no_mv.corr()

"""# 3. Plot correlation matrix"""

# Drop irrelevant columns for correlation analysis
df_new = data.drop(['xlogp', 'exactmass', 'monoisotopicmass'], axis=1)
correlation_matrix = df_new.corr()

# Create the correlation matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Create the correlation matrix heatmap
sns.heatmap(df_new.corr());

# Visualize pairplots for the correlation matrix
sns.pairplot(df_new);

"""# 4. Run the regression"""

y = data_no_mv['xlogp']
X = data_no_mv.drop('xlogp', axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X)

# Display string values in X
string_values = [val for val in X if isinstance(val, str)]
print("String values in X:", string_values)

"""# 5. Display the evaluation metrics"""

import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Decision Tree Regression
regressor = DecisionTreeRegressor()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)

# Display evaluation metric results
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2) Score: {r2}')
print(f'Explained Variance Score: {explained_variance}')
print()

# Visualize the results (scatter plot)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=y_test, y=y_test)
plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Predicted count', fontsize=14)
plt.title('Decision Tree: Actual vs Predicted count (test set)', fontsize=17)
plt.show()

"""# 6. Default parameter"""

from sklearn.tree import DecisionTreeRegressor

# Create an instance of the regression model
decision_tree = DecisionTreeRegressor()

# Get default hyperparameters
default_decision_tree_params = decision_tree.get_params()

# Print the default hyperparameters
print("\nDefault Hyperparameters for Decision Tree Regression:")
print(default_decision_tree_params)

"""# 7. Feature selection ANOVA and Recursive Feature Elimination (RFE)

## ANOVA
"""

# Commented out IPython magic to ensure Python compatibility.

# %matplotlib inline
import statsmodels.api as sm
from statsmodels.formula.api import ols

data_no_mv.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_no_mv' is your DataFrame
plt.figure(figsize=(12, 8))

# Line plot for polararea
sns.lineplot(x='polararea', y='xlogp', data=data_no_mv, label='polararea')

# Line plot for heavycnt
sns.lineplot(x='heavycnt', y='xlogp', data=data_no_mv, label='heavycnt')

# Line plot for hbonddonor
sns.lineplot(x='hbonddonor', y='xlogp', data=data_no_mv, label='hbonddonor')

# Line plot for hbondacc
sns.lineplot(x='hbondacc', y='xlogp', data=data_no_mv, label='hbondacc')

# Line plot for rotbonds
sns.lineplot(x='rotbonds', y='xlogp', data=data_no_mv, label='rotbonds')

plt.title('Line Plots of xlogp based on Categorical Variables')
plt.xlabel('Categories')
plt.ylabel('xlogp Values')
plt.legend(title='Categorical Variable')
plt.show()

model = ols('xlogp ~ polararea + heavycnt + hbonddonor + hbondacc + rotbonds', data=data).fit()
anova = sm.stats.anova_lm(model, typ=2)
anova

"""In ANOVA analysis, the most important or significant parameters are typically identified by the values of F and p (PR(>F)). The higher the F value and the lower the p value, the more significant the influence. In this case: The variables **rotbonds, heavycnt, and polararea** have high F values and very low p values, indicating that these three variables have a significant influence in explaining variability in the data. These values suggest that there is a significant difference between the tested groups for these variables.

"""

# Run the regression

data_new = data_no_mv.drop(['hbondacc', 'hbonddonor'], axis=1)

y_new = data_new['xlogp']
X_new = data_new.drop('xlogp', axis=1)

# Decision Tree Regression
regressor = DecisionTreeRegressor()

X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)

# Display evaluation metric results
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2) Score: {r2}')
print(f'Explained Variance Score: {explained_variance}')
print()

# Visualize the results (scatter plot)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=y_test, y=y_test)
plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Predicted count', fontsize=14)
plt.title('Decision Tree: Actual vs Predicted count (test set)', fontsize=17)
plt.show()

"""## Recursive Feature Elimination (RFE)"""

# Import necessary libraries
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Create the RFE model with Linear Regression
rfe = RFE(estimator=LinearRegression(), n_features_to_select=3)
rfe.fit(X, y)

# Print selected features and their rankings
for i, col in zip(range(len(X.columns)), X.columns):
    print(f"{col} selected={rfe.support_[i]} rank={rfe.ranking_[i]}")

# Plotting the bar graph for feature rankings
plt.figure(figsize=(10, 6))
plt.bar(range(len(rfe.ranking_)), rfe.ranking_)
plt.xticks(range(len(X.columns)), X.columns, rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Ranking')
plt.title('Feature Rankings from RFE')
plt.show()

"""In the realm of Recursive Feature Elimination (RFE) analysis, the significance of features is gauged through their respective rankings. A lower rank denotes higher importance for predicting model outcomes. In this RFE exploration, **polararea, heavycnt, and rotbonds** all share the coveted rank of 1, underscoring their paramount importance in shaping the model's predictive prowess."""

# Run the regression

data_new = data_no_mv.drop(['hbondacc', 'polararea'], axis=1)

y_new1 = data_new['xlogp']
X_new1 = data_new.drop('xlogp', axis=1)

# Decision Tree Regression
regressor = DecisionTreeRegressor()

X_train, X_test, y_train, y_test = train_test_split(X_new1, y_new1, test_size=0.2, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)

# Display evaluation metric results
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2) Score: {r2}')
print(f'Explained Variance Score: {explained_variance}')
print()

# Visualize the results (scatter plot)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=y_test, y=y_test)
plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Predicted count', fontsize=14)
plt.title('Decision Tree: Actual vs Predicted count (test set)', fontsize=17)
plt.show()
