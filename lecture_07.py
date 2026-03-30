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
print(dataset.info())
print(dataset.describe())


#Here implement matplot part

#Seaborn mostly used here ->Seaborn is builted on the top of Matplolib it provide different colours combinationa
# and it has dataset-oriented approach to plotting   which is able to work on actual pandas dataframes

color_palate1 = sns.color_palette("Reds")
sns.palplot(color_palate1)
plt.show()

#for clor blinds
sns.palplot(sns.color_palette("colorblind"))
plt.show()
