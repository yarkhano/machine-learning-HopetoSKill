#Exploratory Data Analysis
#EDA-> Understanding the data , different columns relation etc. Mainly use graphs to show relation ships

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

#Here The madam has loaded data from a file while i loaded from sns library

dataset = sns.load_dataset("titanic")
print(dataset.head())