import pandas as pd
from sklearn.model_selection import train_test_split

dataset=pd.read_csv("C:\\Users\\B!ade\\Downloads\\expdata.csv")

dataset_corr = dataset.corr()

dataset_final = dataset[["Overall", "International Reputation", "Reactions", "Value", "Wage"]]
dataset_final.info()

dataset_final = dataset_final.iloc[:, :].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",axis = 0)
imputer = imputer.fit(dataset_final[:, :])
dataset_final[:, :] = imputer.transform(dataset_final[:, :])

X = dataset_final[:, 0:4]
y = dataset_final[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


####################################################################################################################################################

# Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
y_pred_decision_tree = tree_reg.predict(X_test)


# Residual Squared Error
from sklearn.metrics import r2_score
result_score_tree_reg = r2_score(y_test, y_pred_decision_tree)
print('result score for decision tree  is ', result_score_tree_reg)
####################################################################################################################################################

# Bagging Regressor

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bag_reg = BaggingRegressor(DecisionTreeRegressor(),n_estimators = 100,bootstrap = True,n_jobs = -1)

bag_reg.fit(X_train, y_train)

y_pred = bag_reg.predict(X_test)

# Residual Squared Error
from sklearn.metrics import r2_score
result_score_bag_reg = r2_score(y_test, y_pred)
print('result score for bagging regression is ', result_score_bag_reg)


####################################################################################################################################################

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rnd_reg = RandomForestRegressor()
rnd_reg.fit(X_train, y_train)

y_pred_rf = rnd_reg.predict(X_test)

# Residual Squared Error
from sklearn.metrics import r2_score
result_score_random_reg = r2_score(y_test, y_pred_rf)
print('result score for random forest regression is ', result_score_random_reg)


####################################################################################################################################################

# Linear Regression

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_linear = lin_reg.predict(X_test)

result_score_linear_prediction = r2_score(y_test, y_pred_linear)

print('result score for linear regression is ', result_score_linear_prediction)

####################################################################################################################################################

# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
y_pred_poly = lin_reg_2.predict(poly_reg.fit_transform(X_test))

result_score_polynomial_prediction = r2_score(y_test, y_pred_linear)

print('result score for polynomial regression is ', result_score_polynomial_prediction)
