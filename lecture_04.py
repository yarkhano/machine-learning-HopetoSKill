#Data Preprocessing part 1, This file include code of lecture 04 as well as the code i have added by myself.
#Data cleaning ,outliers detection,finding missing values.


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

dataset = sns.load_dataset("titanic")
print(dataset.shape)
print(dataset.head(10))
print(dataset.describe()) #this give statistical summary of numerica columns(mean,min max)

#Starting on preprocessing
print(dataset.isnull().sum())

print(dataset.dtypes)

#filling missing valuses
dataset["age"] = dataset["age"].fillna(dataset["age"].mean())
dataset["embarked"] = dataset["embarked"].fillna(dataset["embarked"].mode()[0])

#dropping deck column because it has 77% of data missing
dataset.drop(columns=["deck"],inplace=True)
dataset.drop(columns=["embark_town"],inplace=True)
print(dataset.isnull().sum())


#label and one hot encoding
label_encoder = LabelEncoder()
dataset["sex"] = label_encoder.fit_transform(dataset["sex"])
print(dataset.head(10))

dataset = pd.get_dummies(dataset,columns=["embarked"],dtype=int)
print(dataset.head(10))

#Stared feature scalling using standardization as it has outliers
scaler = StandardScaler()

cols = ["age","fare","sibsp", "parch"]
dataset[cols] = scaler.fit_transform(dataset[cols])
print(dataset.head(10))
print(dataset.describe())

#Starting outlier detection using iqr method to catch more outliers, while z score catch extreme outliers
q1 = dataset["fare"].quantile(0.25)
q3 = dataset["fare"].quantile(0.75)
dq = q3-q1

lower_boundary = q1-1.5*dq
upper_boundary = q3+1.5*dq

print("q1:", q1)
print("q3:", q3)
print("IQR:", dq)
print("Lower Boundary:", lower_boundary)
print("Upper Boundary:", upper_boundary)


outliers = dataset[(dataset["fare"]<lower_boundary) | (dataset["fare"]>upper_boundary)]
print("Number of outliers:", len(outliers))

#we have two choices for outliers remove it if not real and have no effect or capping in which we replace outliers with bounderis,using clip()-> anything below lower boundary become value of  lower boundary and same for upper values
#capping outliers
dataset["fare"] = dataset["fare"].clip(lower=lower_boundary, upper=upper_boundary)
outliers_after = dataset[(dataset["fare"]<lower_boundary) | (dataset["fare"]>upper_boundary)]
print("Number of outliers after clipping:", len(outliers_after))