# -*- coding: utf-8 -*-
"""Support Vector Machines (SVM) Regression

# 1. Load a dataset
"""

# Importing the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load your dataset here
# Example: df = pd.read_csv('your_dataset.csv')

# Data Preprocessing

numerical_data = df.drop(df.columns[[0, 1, 2, 3, 4]], axis=1)

# Descriptive statistics for numerical variables
numerical_data.describe(include='all')

# Handling missing values
numerical_data.isnull().sum()

# Drop missing values
data_no_mv = numerical_data.dropna(axis=0)

data_no_mv.isnull().sum()

data_no_mv.describe(include='all')

data_no_mv.info()

cols = [0, 1, 2, 3, 4, 5]

data_no_mv.head()

sns.set_style('whitegrid')
sns.pairplot(data_no_mv[cols])

data_no_mv.corr()

"""#3. Plot correlation matrix"""

df_new = df.drop(df.columns[[0, 1, 2, 4, 5]], axis=1)

# Compute the correlation matrix
correlation_matrix = df_new.corr()

# Create the correlation matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

import seaborn as sns

sns.heatmap(df_new.corr());

sns.pairplot(df_new);

"""# 4. Run the regression

## Scaling the features

### Scaling the variables is a very important step in SVM. Because any variable on the larger scale, has a larger effect on the distance between observations.

### For this dataset, we are going to use standardization as our scaling strategy.
"""

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
df_sc = scale.fit_transform(data_no_mv)
df_sc = pd.DataFrame(df_sc, columns=data_no_mv.columns)

df_sc.head()

"""### Defining the variables and splitting the data"""

y = data_no_mv.iloc[:, 0]
X = data_no_mv.drop(data_no_mv.columns[0], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=365)

# SVM Regression with Sklearn

from sklearn.svm import SVR

# Fitting SVM regression to the training set
SVM_regression = SVR()
SVM_regression.fit(x_train, y_train)

# Predicting the test set result
y_hat = SVM_regression.predict(x_test)

# Display the evaluation metrics

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the models
models = [
    ('Support Vector Regression', SVR())
]
sns.scatterplot(x=y_test, y=y_hat, alpha=0.6)
sns.lineplot(x=y_test, y=y_test)

# Calculating evaluation metrics
mse = mean_squared_error(y_test, y_hat)
mae = mean_absolute_error(y_test, y_hat)
r2 = r2_score(y_test, y_hat)
explained_variance = explained_variance_score(y_test, y_hat)

# Displaying evaluation metric results
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2) Score: {r2}')
print(f'Explained Variance Score: {explained_variance}')
print()

plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Prediced  count', fontsize=14)
plt.title('Actual vs Predicted  count (test set)', fontsize=17)
plt.show()

SVM_regression.score(x_test, y_test)

MSE_test = round(np.mean(np.square(y_test - y_hat)), 2)
MSE_test = round(np.sqrt(MSE_test), 2)
MSE_test

"""## Try to use C = 1

# 6. Tuning hyperparameters:
## Gridsearch

### Finding the right hyper parameters (like C, gamma and the Kernel function) is a tricky task! But luckily, we can be a little lazy and just try a bunch of combinations and see what works best! This idea of creating a 'grid' of parameters and just trying out all the possible combinations is called a Gridsearch, this method is common enough that Scikit-learn has this functionality built in with GridSearchCV! The CV stands for cross-validation.

GridSearchCV takes a dictionary that describes the parameters that should be tried and a model to train. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested.

C represents cost of misclassification. A large C means that you are penalizing the errors more restricly so the margin will be narrower ie overfitting (small bias, big variance) https://scikit-learn.org/stable/modules/svm.html

gamma is the free parameter in the radial basis function (rbf). Intuitively, the gamma parameter (inverse of variance) defines how far the influence of a single training example reaches with low values meaning ‘far’ and high values meaning ‘close’.

https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
"""

my_param_grid = {'C': [1, 10, 100], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV

GridSearchCV(estimator=SVR(), param_grid=my_param_grid, refit=True, verbose=3, cv=5)

grid = GridSearchCV(estimator=SVR(), param_grid=my_param_grid, refit=True, verbose=2, cv=5)
# verbose just means the text output describing the process. (the greater the number the more detail you will get).

# May take a while!
grid.fit(x_train, y_train)

"""### What fit does is a bit more involved than usual. First, it tries multiple combinations from param_grid by cross-validation to find the best parameter combination. Once it has the best combination, it retrain the model using optimal hyperparameters on the entire train set."""

grid.best_params_

grid.best_estimator_

y_hat_optimized = grid.predict(x_test)

# Display the evaluation metrics

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the models
models = [
    ('Support Vector Regression', SVR())
]

sns.scatterplot(x=y_test, y=y_hat_optimized, alpha=0.6)
sns.lineplot(x=y_test, y=y_test)

# Calculating evaluation metrics
mse = mean_squared_error(y_test, y_hat)
mae = mean_absolute_error(y_test, y_hat)
r2 = r2_score(y_test, y_hat)
explained_variance = explained_variance_score(y_test, y_hat)

# Displaying evaluation metric results
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2) Score: {r2}')
print(f'Explained Variance Score: {explained_variance}')
print()

plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Prediced  count', fontsize=14)
plt.title('Actual vs Predicted  count (test set)', fontsize=17)
plt.show()

# sns.scatterplot(x=y_test, y=y_hat_optimized, alpha=0.6)
# sns.lineplot(x=y_test, y=y_test)

# plt.xlabel('Actual count', fontsize=14)
# plt.ylabel('Prediced  count', fontsize=14)
# plt.title('Actual vs optimized predicted count (test set)', fontsize=17)
# plt.show()

grid.score(x_test, y_test)

MSE_test_opt = round(np.mean(np.square(y_test - y_hat_optimized)), 2)
RMSE_test_opt = round(np.sqrt(MSE_test_opt), 2)
RMSE_test_opt

"""## Cross validation
### We will use Cross validation to estimate performance metrics in the test set.
"""

from sklearn.model_selection import cross_val_score

NMSE = cross_val_score(estimator=SVR(C=1, gamma=0.01), X=x_train, y=y_train, cv=5, scoring="neg_mean_squared_error")

MSE_CV = round(np.mean(-NMSE), 4)
RMSE_CV = round(np.sqrt(MSE_CV), 4)
RMSE_CV

"""#7. Feature selection ANOVA and Recursive Feature Elimination (RFE)

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
sns.lineplot(x=2, y=0, data=data_no_mv, label='polararea')

# Line plot for heavycnt
sns.lineplot(x=3, y=0, data=data_no_mv, label='heavycnt')

# Line plot for hbonddonor
sns.lineplot(x=4, y=0, data=data_no_mv, label='hbonddonor')

# Line plot for hbondacc
sns.lineplot(x=5, y=0, data=data_no_mv, label='hbondacc')

# Line plot for rotbonds
sns.lineplot(x=6, y=0, data=data_no_mv, label='rotbonds')

plt.title('Line Plots of xlogp based on Categorical Variables')
plt.xlabel('Categories')
plt.ylabel('xlogp Values')
plt.legend(title='Categorical Variable')
plt.show()

model = ols('0 ~ 2 + 3 + 4 + 5 + 6', df).fit()
anova = sm.stats.anova_lm(model, typ=2)
anova

"""In ANOVA analysis, the most important or significant parameters are typically identified by the values of F and p (PR(>F)). The higher the F value and the lower the p value, the more significant the influence. In this case: The variables **rotbonds, heavycnt, and polararea** have high F values and very low p values, indicating that these three variables have a significant influence in explaining variability in the data. These values suggest that there is a significant difference between the tested groups for these variables.

"""

# Run the regression

data_new = data_no_mv.drop([5, 4], axis=1)

y_new = data_new.iloc[:, 0]
x_new = data_new.drop(data_new.columns[0], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=365)

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import seaborn as sns
import matplotlib.pyplot as plt


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=365)

# Define the SVR model with modified parameters
svr_model = SVR(kernel='linear', C=1.0, gamma='scale')

# Train the SVR model
svr_model.fit(x_train, y_train)

# Make predictions
y_hat = svr_model.predict(x_test)

# Plotting the scatterplot and lineplot
sns.scatterplot(x=y_test, y=y_hat, alpha=0.6)
sns.lineplot(x=y_test, y=y_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_hat)
mae = mean_absolute_error(y_test, y_hat)
r2 = r2_score(y_test, y_hat)
explained_variance = explained_variance_score(y_test, y_hat)

# Display evaluation metric results
print(f'Support Vector Regression:\nMean Squared Error (MSE): {mse}\nMean Absolute Error (MAE): {mae}\nR-squared (R2) Score: {r2}\nExplained Variance Score: {explained_variance}\n')

# Plotting settings
plt.xlabel('Actual xlogp', fontsize=14)
plt.ylabel('Predicted xlogp', fontsize=14)
plt.title('Actual vs Predicted xlogp (test set)', fontsize=17)
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

# Plotting the bar graph for feature rankings
plt.figure(figsize=(10, 6))
plt.bar(range(len(rfe.ranking_)), rfe.ranking_)
plt.xticks(range(len(X.columns)), X.columns, rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Ranking')
plt.title('Feature Rankings from RFE')
