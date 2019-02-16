'''
Using Multiple Linear Regression by backward Elimination#
This script will have additional steps for backward elimination to chose the best independent variables.
By best, we mean the one which have more impact in predicting the output.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utility.utils as utl
#import utility.utils as utl
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
#you might want to check for categories='auto'
onehotEnc = OneHotEncoder(categorical_features=[3])
X = onehotEnc.fit_transform(X).toarray()

#Avoid the dummy variable Trap, remove one dummy variable column
#Selecting all column starting from index - 1, In most cases libraries take care of it
X = X[:,1:]


#divide X,Y into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=0)

#fitting the data into MLR

from sklearn.linear_model import LinearRegression
startup_mlr = LinearRegression()

startup_mlr.fit(X_train,y_train)

#model created, lets test the model on test data
y_pred_test = startup_mlr.predict(X_test)
'''
utl.print_actual_predicted(y_test,y_pred_test,True)

#model created, lets test the model on training data
y_pred_train = startup_mlr.predict(X_train)

utl.print_actual_predicted(y_train,y_pred_train,True)
'''

#building the optimal model via backward elimination
import statsmodels.formula.api as sm
#add column of 1, to make y = b0X0+b1X1....+bnXn. Here X0=1
#Step 0 - add 1 column for bo
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
#Step 1 - set SL = 0.5 (Significance level)
SL = 0.5

#Step 2 - fit all the predictor in the model
#create a new matrix which will contain the optimal features (right now all of them)
X_optimal = X[:,[0,1,2,3,4,5]]

#create the regression with least square from statsmodels.formula.api
#for using this we have added 1 column in Step 0  as the library will not take the intercept b0
regressior_OLS = sm.OLS(endog = y,exog=X_optimal).fit()
#This will print the summary from which we can get the P-value of different independet variables
regressior_OLS.summary()



#Step 3 - Select the predictor/feature/independent variable with highest p-value 
# check if p-value > S.L. goto step 4 otherwise we got out model with needed predictors.
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     169.9
Date:                Sun, 10 Feb 2019   Prob (F-statistic):           1.34e-27
Time:                        21:01:49   Log-Likelihood:                -525.38
No. Observations:                  50   AIC:                             1063.
Df Residuals:                      44   BIC:                             1074.
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================
Omnibus:                       14.782   Durbin-Watson:                   1.283
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.266
Skew:                          -0.948   Prob(JB):                     2.41e-05
Kurtosis:                       5.572   Cond. No.                     1.45e+06
==============================================================================

'''


#Step 4 - We took X2 p-value = 0.990 which is greater then 0.5
#Remove this predictor at index 2
X_optimal = X[:,[0,1,3,4,5]]


#Step 5 - fit the model again
regressior_OLS = sm.OLS(endog = y,exog=X_optimal).fit()

#Step 3
regressior_OLS.summary()

'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.946
Method:                 Least Squares   F-statistic:                     217.2
Date:                Sun, 10 Feb 2019   Prob (F-statistic):           8.49e-29
Time:                        21:21:30   Log-Likelihood:                -525.38
No. Observations:                  50   AIC:                             1061.
Df Residuals:                      45   BIC:                             1070.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
==============================================================================
Omnibus:                       14.758   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.172
Skew:                          -0.948   Prob(JB):                     2.53e-05
Kurtosis:                       5.563   Cond. No.                     1.40e+06
==============================================================================
'''

#Now x1 is p-value = .94 > SL, repeat step 4,5 and go to step 3
#Step 4
X_optimal = X[:,[0,3,4,5]]
#Step 5 - fit the model again
regressior_OLS = sm.OLS(endog = y,exog=X_optimal).fit()
#Step 3
regressior_OLS.summary()

'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     296.0
Date:                Sun, 10 Feb 2019   Prob (F-statistic):           4.53e-30
Time:                        21:24:03   Log-Likelihood:                -525.39
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      46   BIC:                             1066.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================
Omnibus:                       14.838   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.442
Skew:                          -0.949   Prob(JB):                     2.21e-05
Kurtosis:                       5.586   Cond. No.                     1.40e+06
==============================================================================
'''

#Now x2 is p-value = .602 > SL, repeat step 4,5 and go to step 3
#Step 4
X_optimal = X[:,[0,3,5]]
#Step 5 - fit the model again
regressior_OLS = sm.OLS(endog = y,exog=X_optimal).fit()
#Step 3
regressior_OLS.summary()

'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.950
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     450.8
Date:                Sun, 10 Feb 2019   Prob (F-statistic):           2.16e-31
Time:                        21:25:30   Log-Likelihood:                -525.54
No. Observations:                  50   AIC:                             1057.
Df Residuals:                      47   BIC:                             1063.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
Omnibus:                       14.677   Durbin-Watson:                   1.257
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.161
Skew:                          -0.939   Prob(JB):                     2.54e-05
Kurtosis:                       5.575   Cond. No.                     5.32e+05
==============================================================================
'''


#Now x2 is p-value = .06 > SL, repeat step 4,5 and go to step 3
X_optimal = X[:,[0,3]]
#Step 5 - fit the model again
regressior_OLS = sm.OLS(endog = y,exog=X_optimal).fit()
#Step 3
regressior_OLS.summary()

'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.947
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     849.8
Date:                Sun, 10 Feb 2019   Prob (F-statistic):           3.50e-32
Time:                        21:34:35   Log-Likelihood:                -527.44
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      48   BIC:                             1063.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
==============================================================================
Omnibus:                       13.727   Durbin-Watson:                   1.116
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               18.536
Skew:                          -0.911   Prob(JB):                     9.44e-05
Kurtosis:                       5.361   Cond. No.                     1.65e+05
==============================================================================
'''

#So the R&D value is most important predicotr/independent 
print("New Optimal Input")
print(X_optimal)

print("Creating MLR using the new optimal set")

#divide X,Y into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_optimal,y,test_size=1/5,random_state=0)

#fitting the data into MLR

from sklearn.linear_model import LinearRegression
startup_mlr = LinearRegression()

startup_mlr.fit(X_train,y_train)

#model created, lets test the model on test data
y_pred_test = startup_mlr.predict(X_test)

utl.print_actual_predicted(y_test,y_pred_test,True)

#model created, lets test the model on training data
y_pred_train = startup_mlr.predict(X_train)

utl.print_actual_predicted(y_train,y_pred_train,True)