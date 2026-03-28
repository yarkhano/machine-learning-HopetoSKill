#linear regression

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
print(df.head())

X = dataset.data
y = dataset.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(y_pred)

# model coefficients m and b
print("Model Coefficients")
print(f"slope:{model.coef_[0]}")
print(f"intercept:{model.intercept_}")