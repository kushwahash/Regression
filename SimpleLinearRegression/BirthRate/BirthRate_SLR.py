#Data preprocessing Start


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


#import the SLR library
from sklearn.linear_model import LinearRegression
slr_birth = LinearRegression()
#making it 2d array
X_train = X_train.reshape(-1,1)
slr_birth.fit(X_train,y_train)

#use the model created to predict the test data.
y_pred_train = slr_birth.predict(X_train)

print("++++ Running model on Training Data ++++")
for i in range(0,len(y_train)):
    print("Actual :: {}, Predicted :: {:.1f}".format(y_train[i],y_pred_train[i]))

#Run the model on test set
X_test = X_test.reshape(-1,1)
y_pred_test = slr_birth.predict(X_test)
print("++++ Predicting Test Data +++++")
for i in range(0,len(y_test)):
    print("Actual :: {}, Predicted :: {:.1f}".format(y_train[i],y_pred_test[i]))

#visualize the data and the predicted line train data
plt.scatter(X_train,y_train,color = 'red')
#visualize the Train this will show the line which is predicted
plt.plot(X_train,slr_birth.predict(X_train),color = 'blue')
#add a Title to better understand
plt.title('Povert vs Teen Birth Rate (Age15to17) Training Data Set')
plt.xlabel('Poverty')
plt.ylabel('Teen Birth Rate')
plt.show()

#visualize the data and the predicted line test data
plt.scatter(X_test,y_test,color = 'red')
#visualize the Train this will show the line which is predicted
plt.plot(X_train,slr_birth.predict(X_train),color = 'blue')
#add a Title to better understand
plt.title('Povert vs Teen Birth Rate (Age15to17) Test Data Set')
plt.xlabel('Poverty')
plt.ylabel('Teen Birth Rate')
plt.show()

