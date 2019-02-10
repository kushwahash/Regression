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