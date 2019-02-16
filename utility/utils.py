'''
Will create different util functions which are getting repeated.
'''

def print_actual_predicted(actual,predicted,show_error_precentage):
    '''
    Description : A function to print the actual and predicted data side by side for comparison.
    Input ::
        actual : 1d Array of actual data.
        predicted : 1d Array of predicted data.
        show_error_precentage: Whether we want to see the error percentage of our prediction.
    Note : Here the predicted data is the driving force and we will print the number of items
            found in the predicted array.
    '''
    print("\n\n++++ Actual vs Predicted Data +++++\n\n")
    if show_error_precentage:
        total_error = 0
        for i in range(0,len(predicted)):
            error_rate = (abs(actual[i]-predicted[i])/actual[i])*100
            total_error += error_rate
            print("Actual :: {:<10} Predicted :: {:<10.1f}, Error :: {:<10.2f}".format(actual[i],predicted[i],error_rate))
        print("\n\nNumber of cases :: {}, Total Error Percentage :: {:.2f}\n\n".format(len(predicted),total_error/len(predicted)))
    else:
        for i in range(0,len(predicted)):
            print("Actual :: {:<10}, Predicted :: {:<10.1f}".format(actual[i],predicted[i]))

'''
import statsmodels.formula.api as sm
import numpy as np
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
'''