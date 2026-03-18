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