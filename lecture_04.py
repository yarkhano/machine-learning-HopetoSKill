#Data Preprocessing part 1, This file include code of lecture 04 as well as the code i have added by myself
#Data cleaning ,outliers detection,finding missing values.

import pandas as pd
import seaborn as sns

dataset = sns.load_dataset("titanic")
print(dataset.shape)

