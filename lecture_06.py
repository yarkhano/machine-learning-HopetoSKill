#linear regression

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

#drawing plots using matplotlib
single_feature = X_test[:,0]
y_pred = model.predict(X_test)

plt.scatter(single_feature,y_test,color='blue',label='Actual')
plt.scatter(single_feature,y_pred,color='red',label='Predicted')
plt.xlabel('Medinc')
plt.ylabel("Housing Price")
plt.legend()
plt.show()
# model coefficients m and b
print("Model Coefficients")
print(f"slope:{model.coef_[0]}")
print(f"intercept:{model.intercept_}")