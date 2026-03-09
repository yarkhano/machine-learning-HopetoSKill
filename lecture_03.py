#Attributes in Data

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
attributes = iris.feature_names
targets = iris.target

print(attributes)
print(targets)

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(df)

df["target"] = iris.target

print(df.head())

print(df.dtypes)

print(df.shape) #tell how many rows and columns we have