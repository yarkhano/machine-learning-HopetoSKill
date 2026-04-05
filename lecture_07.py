#Exploratory Data Analysis
#EDA-> Understanding the data , different columns relation etc. Mainly use graphs to show relation ships

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly.express as px

#Here The madam has loaded data from a file while i loaded from sns library

dataset = sns.load_dataset("titanic")
# print(dataset.head())
# print(dataset.info())
# print(dataset.describe())
#
#
# #Here implement matplot part
#
# #Seaborn mostly used here ->Seaborn is builted on the top of Matplolib it provide different colours combinationa
# # and it has dataset-oriented approach to plotting   which is able to work on actual pandas dataframes
#
# color_palate1 = sns.color_palette("Reds")
# sns.palplot(color_palate1)
# plt.show()
#
# #for color blinds
# sns.palplot(sns.color_palette("colorblind"))
# plt.show()
#
#
# #Using matplotlib
# plt.figure(figsize=(10,6))
# sns.countplot(x='pclass',hue='survived',data=dataset)
# plt.xlabel('Pclass')
# plt.ylabel('Survived')
# plt.legend(labels = ['Did not survive', 'Survived'])
# plt.show()
#
# plt.figure(figsize=(10,6))
# sns.countplot(x='sex',hue='survived',data=dataset)
# plt.title('Survived w.r.t Sex')
# plt.xlabel('sex')
# plt.ylabel('count')
# plt.legend(labels=['Did Not Survive','Survived'])
# plt.show()
#
# #Using kde=True to show curve for average value.
# plt.figure(figsize=(10,6))
# sns.histplot(data=dataset,x='age',kde=True)
# plt.title('Survived w.r.t Sex')
# plt.xlabel('Age')
# plt.ylabel('count')
# plt.axvline(dataset["age"].mean(), color='red', linestyle='--',label='Mean Age' )
# plt.legend()
# plt.show()
#
# #Facet Grid
# g = sns.FacetGrid(dataset,col='survived',row='pclass')
# g.map(plt.hist,'age')
# g.add_legend()
# plt.show()

#Starting plotly.express
fig1 = px.bar(dataset,x="survived",title="Survival Count",labels={"survived":"survived (0=N0,1=Yes)"})
fig1.show()