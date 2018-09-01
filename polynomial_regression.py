# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Create model, fit and train with simple linear regression
from sklearn.linear_model import LinearRegression
linreg_1 = LinearRegression()
linreg_1.fit(X,y)

#transform variables to ploynomial and fit them 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

linreg_2 = LinearRegression()
linreg_2.fit(X_poly,y)

#Visualising the result with linear regression 
plt.scatter(X,y, color ="red")
plt.plot(X,linreg_1.predict(X),color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#Visualising the result with polynomial regression 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color ="red")
plt.plot(X,linreg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predict new result with linear regression 
linreg_1.predict(6.5)
#predict new result with polynomial regression
linreg_2.predict(poly_reg.fit_transform(6.5))