#Attributes in Data
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
attributes = data.feature_names
targets = data.target

print(attributes)
print(targets)

df = pd.DataFrame(data=data.data, columns=data.feature_mnames)
print(df)

df["target"] = data.target
