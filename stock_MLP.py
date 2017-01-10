import numpy as np
from TFMLP import MLPR
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor

pth = '/home/lanhnguyen/workspace/pythonml-master/yahoo.csv'
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1, 5))
x_mean= np.mean(A[:,0])
y_mean= np.mean(A[0,:])
A = scale(A)

#y is the dependent variable
y = A[:, 1].reshape(-1, 1)
#A contains the independent variable
A = A[:, 0].reshape(-1, 1)
#Plot the high value of the stock price
# mpl.plot(A[:, 0], y[:, 0])
# mpl.show()
#Number of neurons in the input layer
i = 1
#Number of neurons in the output layer
o = 1
#Number of neurons in the hidden layers
h = 32
#The list of layer sizes
layers = [i, h, h, o]
# mlpr = KNeighborsRegressor(n_neighbors = 5)
mlpr = MLPR(layers, maxItr = 100000, tol = 0.4, reg = 0.001, verbose = True)
#Length of the hold-out period
nDays = 200
n = len(A)
#Learn the data
mlpr.fit(A[0:(n-nDays)], y[0:(n-nDays)])
#Begin prediction
print("predict data")
yHat = mlpr.predict(A)
score = mlpr.score(A, y)
#Plot the results
mpl.plot(A, y, c='#b0403f', label="Actual")
mpl.plot(A, yHat, c='g', label="Predicted")
mpl.ylabel('Stock One Day Prediction')
print(score)
mpl.show()