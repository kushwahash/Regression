#Using Multiple Linear Regression#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#read the file
dataset = pd.read_csv("50_Startups.csv")
#set the independent and dependent variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values


#Encoding categorical data (State)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEnc_X = LabelEncoder()
#country is at index 3
X[:,3] = labelEnc_X.fit_transform(X[:,3])
onehotEnc = OneHotEncoder(categorical_features=[3])
X = onehotEnc.fit_transform(X).toarray()

#Avoid the dummy variable Trap, remove one dummy variable column
#Selecting all column starting from index - 1, In most cases libraries take care of it
X = X[:,1:]


#divide X,Y into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=0)




