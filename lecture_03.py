#Attributes in Data
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
attributes = data.feature_names
targets = data.target

print(attributes)
print(targets)