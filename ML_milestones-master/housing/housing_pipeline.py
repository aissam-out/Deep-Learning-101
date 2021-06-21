import pandas as pd
import numpy as np
import someFunctions as smfcts
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

housingAll = pd.read_csv("housing.csv")

#split the housingAll to train and test set
train_set, test_set = train_test_split(housingAll, test_size=0.2, random_state=42)

#prepare the x and y
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

#fill na data
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1) #keep just the numerical data
imputer.fit(housing_num)
x = imputer.transform(housing_num) #x is a plain Numpy array containing the transformed features by imputer
housing_tr = pd.DataFrame(x, columns=housing_num.columns)

#StandardScaler
scaler = StandardScaler()
housing_scaled = scaler.fit_transform(housing_tr) #this is a plain Numpy array
housing_scaled_df = pd.DataFrame(housing_scaled, columns=housing_num.columns)

#train a model
lin_reg = LinearRegression()
lin_reg.fit(housing_scaled_df, housing_labels)

#testing some instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
#preparation
some_data_num = some_data.drop("ocean_proximity", axis=1)
imputer.fit(some_data_num)
ex = imputer.transform(some_data_num)
some_data_tr = pd.DataFrame(ex, columns=some_data_num.columns)
some_data_scaled = scaler.fit_transform(some_data_tr)
some_data_scaled_df = pd.DataFrame(some_data_scaled, columns=some_data_num.columns)
#test
print("predictions:", lin_reg.predict(some_data_scaled_df).astype(int)) #astype to convert the float values to int ones
print("real labels:", list(some_labels))

#evaluation
housing_predictions = lin_reg.predict(housing_scaled_df)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Evaluation of linear regression learning: RMSE =", lin_rmse)

#try decision tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_scaled_df, housing_labels)
housing_predictions_2 = tree_reg.predict(housing_scaled_df)
tree_mse = mean_squared_error(housing_labels, housing_predictions_2)
tree_rmse = np.sqrt(tree_mse)
print("Evaluation of decision tree learning: RMSE =", tree_rmse)

#(k-fold) cross validation
#cross validation of decision tree
scores = cross_val_score(tree_reg, housing_scaled_df, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
smfcts.display_scores(tree_rmse_scores)
#cross validation of linear regression
lin_scores = cross_val_score(lin_reg, housing_scaled_df, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
smfcts.display_scores(lin_rmse_scores)

#try random forest
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_scaled_df, housing_labels)
housing_predictions_3 = forest_reg.predict(housing_scaled_df)
forest_mse = mean_squared_error(housing_labels, housing_predictions_3)
forest_rmse = np.sqrt(forest_mse)
print("Evaluation of random forest learning: RMSE =", forest_rmse)
forest_scores = cross_val_score(forest_reg, housing_scaled_df, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
smfcts.display_scores(forest_rmse_scores)
