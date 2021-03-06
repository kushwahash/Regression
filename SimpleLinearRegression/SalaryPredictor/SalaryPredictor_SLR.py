#Data preprocessing Start

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read the data file
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#split the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)


#we don't need feature scaling as SLR algorithm take care of it
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

#Simple Linear Regression to the Training Set
#import the needed library
from sklearn.linear_model import LinearRegression
slr = LinearRegression()

#fit the training data Independent -X, dependent - y which we have created earlier
#This will create the SLR model which can be used to predict the test set
slr.fit(X_train,y_train)

# Predict the test set result

y_pred = slr.predict(X_test)

#Visualize (show) the training set results this will show points
plt.scatter(X_train,y_train,color = 'red')
#visualize the Train this will show the line which is predicted
plt.plot(X_train,slr.predict(X_train),color = 'blue')
#add a Title to better understand
plt.title('Salary vs Age (Experience) Training Set')
plt.xlabel('Age (Experience)')
plt.ylabel('Salary')
plt.show()


#Visualize (show) the test set results this will show points
plt.scatter(X_test,y_test,color = 'red')
#visualize the test this will show the line which is predicted
plt.plot(X_train,slr.predict(X_train) ,color = 'blue')
#add a Title to better understand
plt.title('Salary vs Age (Experience) Training Set')
plt.xlabel('Age (Experience)')
plt.ylabel('Salary')
plt.show()
