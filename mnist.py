import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as d

data = pd.read_csv("train.csv")

#print("Dta\n",data) 

data = pd.read_csv("train.csv").as_matrix()
print("Matrix data\n",data)

clf = d()

xtrain = data[0:21000, 1:]
ytrain = data[0:21000, 0]

xtest = data[21000:, 1:]
ytest = data[21000:, 0]

disp = xtest[8]
disp.shape = (28,28)
plt.imshow(255 - disp, cmap="gray")
plt.show()

p = clf.predict(xtest[8])
print("Prediction=", p)
