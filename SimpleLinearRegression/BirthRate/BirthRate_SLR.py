#Data preprocessing Start

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read the data file
dataset = pd.read_csv("BirthvsPoverty.csv")
#Will take Second Column Poverty Percentage as Independent variable.
X = dataset.iloc[:, 1].values
#Using Third Column,Brth15to17 as Dependent variable. 
y = dataset.iloc[:, 2].values


#split the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=0)

#Visualize (show) the actual data set. We can take a look and see if using SLR is a good idea (Slope)
plt.scatter(X,y,color = 'red')
#add a Title to better understand
plt.title('Poverty vs Teen Birth (Age15to17) Training Set (Actual Data)')
plt.xlabel('Poverty')
plt.ylabel('Teen Birth')
plt.show()