#linear regression
#I have also implemented  some other things not being covered in that lecture like evaluation,feature scalling
#the results before feature scalling was average lets scale it and see it then what will happen
#did all for linear regression but the dataset I have used is complex so inear regression is unable to capture the patterns,np effect of feature scalling

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
print(df.head())

X = dataset.data
y = dataset.target

scaler = StandardScaler()


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)


single_feature = x_test_scaled[:,0]

#Evaluating the model
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)
print(f'MSE: {mse}')
print(f'R2: {r2}')
print(f'RMSE: {rmse}')



#drawing plots using matplotlib
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
