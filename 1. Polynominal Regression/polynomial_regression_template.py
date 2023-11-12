# -*- coding: utf-8 -*-
"""Polynomial Regression Template

This script performs Polynomial Regression on a given dataset after cleaning and feature selection.

# 1. Load a dataset
"""

## IMPORTING THE RELEVANT LIBRARIES
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset here
# Example: data = pd.read_csv('your_dataset.csv')

"""# 2. Clean the data"""

# Check for missing values
data.isnull().sum()

# Drop missing values
numerical_data = data.drop(['column1', 'column2'], axis=1)  # Specify columns to drop
data_no_mv = numerical_data.dropna(axis=0)

# Display descriptive statistics
data_no_mv.describe(include='all')

# Display information about the cleaned data
data_no_mv.info()

# Display columns of the cleaned data
data_no_mv.columns

# Selected columns for analysis
cols = ['target_column', 'feature1', 'feature2']

# Visualize pairplots for selected columns
sns.pairplot(data_no_mv[cols])

# Calculate correlation matrix
data_no_mv.corr()

"""# 3. Plot correlation matrix"""

# Drop irrelevant columns for correlation analysis
df_new = data.drop(['column1', 'column2'], axis=1)  # Specify columns to drop

# Compute the correlation matrix
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

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create a PolynomialFeatures transformer
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

# Transform the original features to include polynomial, interaction, and ratio terms
X_poly = poly.fit_transform(data_no_mv)

# Get the feature names
feature_names = poly.get_feature_names_out(input_features=list(data_no_mv.columns))
X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
X_poly_df

# Scaling
from sklearn.preprocessing import StandardScaler  # Z-score normalization
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly_df)
X_poly_scaled_df = pd.DataFrame(X_poly_scaled, columns=list(X_poly_df.columns))

X_poly_scaled_df

y = data_no_mv['target_column']
X = data_no_mv.drop('target_column', axis=1)

# Modeling
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score

# Create a linear regression model
model = LinearRegression()

# Create an RFECV object with scoring as R2
rfecv = RFECV(estimator=model, cv=10, scoring='r2')  # You can adjust the number of cross-validation folds as needed

# Fit the RFECV model to the scaled data
rfecv.fit(X_poly_scaled_df, y)

# Determine the selected features and their names
selected_features_indices = np.where(rfecv.support_)[0]
selected_feature_names = X_poly_scaled_df.columns[selected_features_indices]
X_selected = X_poly_scaled_df.iloc[:, selected_features_indices]

# Build your final model
final_model = LinearRegression()
final_model.fit(X_selected, y)
y_pred_selected = final_model.predict(X_selected)

# Find coefficients
feature_importances = final_model.coef_
# Pair the feature names with their respective coefficients
feature_importance_dict = dict(zip(selected_feature_names, feature_importances))

print("Selected Features:")
print(selected_feature_names)

# Print or analyze the feature importances
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance:.4f}")

"""# 5. Display the evaluation metrics"""

# Define the models
models = [
    ('Polynomial Regression', PolynomialFeatures(degree=2))
]

# Fit each model, make predictions, and evaluate
for model_name, regressor in models:
    if isinstance(regressor, PolynomialFeatures):
        # For Polynomial Regression, transform the features
        X_poly = regressor.fit_transform(X)
        regressor = SVR()  # Replace with your desired regression model for Polynomial Regression
    else:
        X_poly = X  # For other models, use the original features

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
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

# Scatterplot and lineplot for actual vs predicted values
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=y_test, y=y_test)
plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Predicted count', fontsize=14)
plt.title('Actual vs Predicted count (test set)', fontsize=17)
plt.show()

"""# 6. Default parameter"""

poly_reg = PolynomialFeatures(degree=2)
default_poly_reg_params = poly_reg.get_params()

# Print the default hyperparameters
print("Default Hyperparameters for Polynomial Regression:")
print(default_poly_reg_params)

"""# 7. Feature selection ANOVA and Recursive Feature Elimination (RFE)

## ANOVA
"""

# Commented out IPython magic to ensure Python compatibility.
# ANOVA
# %matplotlib inline
import statsmodels.api as sm
from statsmodels.formula.api import ols

data_no_mv.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_no_mv' is your DataFrame
plt.figure(figsize=(12, 8))

# Line plot for feature1
sns.lineplot(x='feature1', y='target_column', data=data_no_mv, label='feature1')

# Line plot for feature2
sns.lineplot(x='feature2', y='target_column', data=data_no_mv, label='feature2')

# Line plot for feature3
sns.lineplot(x='feature3', y='target_column', data=data_no_mv, label='feature3')

# Line plot for feature4
sns.lineplot(x='feature4', y='target_column', data=data_no_mv, label='feature4')

# Line plot for feature5
sns.lineplot(x='feature5', y='target_column', data=data_no_mv, label='feature5')

plt.title('Line Plots of target_column based on Categorical Variables')
plt.xlabel('Categories')
plt.ylabel('target_column Values')
plt.legend(title='Categorical Variable')
plt.show()

model = ols('target_column ~ feature1 + feature2 + feature3 + feature4 + feature5', data=data).fit()
anova = sm.stats.anova_lm(model, typ=2)
anova

"""In ANOVA analysis, the most important or significant parameters are typically identified by the values of F and p (PR(>F)). The higher the F value and the lower the p value, the more significant the influence. In this case: The variables **feature4, feature3, and feature1** have high F values and very low p values, indicating that these three variables have a significant influence in explaining variability in the data. These values suggest that there is a significant difference between the tested groups for these variables.

"""

# Run the regression

data_new = data_no_mv.drop(['feature5', 'feature2'], axis=1)

y_new = data_new['target_column']
X_new = data_new.drop('target_column', axis=1)

# Define the models
models = [
    ('Polynomial Regression', PolynomialFeatures(degree=2))
]

# Fit each model, make predictions, and evaluate
for model_name, regressor in models:
    if isinstance(regressor, PolynomialFeatures):
        # For Polynomial Regression, transform the features
        X_poly = regressor.fit_transform(X_new)
        regressor = SVR()  # Replace with your desired regression model for Polynomial Regression
    else:
        X_poly = X_new  # For other models, use the original features

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y_new, test_size=0.2, random_state=42)
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

# Scatterplot and lineplot for actual vs predicted values
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=y_test, y=y_test)
plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Predicted count', fontsize=14)
plt.title('Polynomial : Actual vs Predicted count (test set)', fontsize=17)
plt.show()

"""## Recursive Feature Elimination (RFE)"""

# Import necessary libraries
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create the RFE model with Linear Regression
rfe = RFE(estimator=LinearRegression(), n_features_to_select=3)
rfe.fit(X, y)

# Print selected features and their rankings
for i, col in zip(range(len(X.columns)), X.columns):
    print(f"{col} selected={rfe.support_[i]} rank={rfe.ranking_[i]}")

# Plot the bar graph for feature rankings
plt.figure(figsize=(10, 6))
plt.bar(range(len(rfe.ranking_)), rfe.ranking_)
plt.xticks(range(len(X.columns)), X.columns, rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Ranking')
plt.title('Feature Rankings from RFE')
plt.show()

"""In the realm of Recursive Feature Elimination (RFE) analysis, the significance of features is gauged through their respective rankings. A lower rank denotes higher importance for predicting model outcomes. In this RFE exploration, **feature4, feature3, and feature1** all share the coveted rank of 1, underscoring their paramount importance in shaping the model's predictive prowess."""

# Run the regression

data_new = data_no_mv.drop(['feature5', 'feature3'], axis=1)

y_new1 = data_new['target_column']
X_new1 = data_new.drop('target_column', axis=1)

# Define the models
models = [
    ('Polynomial Regression', PolynomialFeatures(degree=2))
]

# Fit each model, make predictions, and evaluate
for model_name, regressor in models:
    if isinstance(regressor, PolynomialFeatures):
        # For Polynomial Regression, transform the features
        X_poly = regressor.fit_transform(X_new1)
        regressor = SVR()  # Replace with your desired regression model for Polynomial Regression
    else:
        X_poly = X_new1  # For other models, use the original features

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y_new1, test_size=0.2, random_state=42)
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

# Scatterplot and lineplot for actual vs predicted values
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=y_test, y=y_test)
plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Predicted count', fontsize=14)
plt.title('Polynomial: Actual vs Predicted count (test set)', fontsize=17)
plt.show()
