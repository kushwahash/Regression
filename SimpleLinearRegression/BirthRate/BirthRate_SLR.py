#Data preprocessing Start

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read the data file
dataset = pd.read_csv("Birth_Rate.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values