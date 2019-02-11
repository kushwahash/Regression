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