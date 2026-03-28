#linear regression

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = fetch_california_housing.data()

X = dataset.data
y = dataset.target

X_train,X_test,y_train,y_tset = train_test_split(X,y,test_size=0.2,random_state=42)