import math
from re import X
import numpy as np
import matplotlib.pyplot as plt
y = np.loadtxt('data/GSPC.csv', delimiter=',', unpack=True, skiprows=1, usecols=4)
x = np.arange(len(y))

xMat = np.matrix(x).T
m = xMat.shape[0]

xCon = np.concatenate(([np.ones((m, 1)), xMat]),1)

def lwr(x, y, tau):
    yPred = np.zeros(m)
    for i in range(m):
        yPred[i] = (x[i] * localWeight(x[i], x, y, tau)).item()
    return yPred

def localWeight(x1, x, y, tau):
    wt = funct(x1, x, tau)
    w = np.linalg.inv(x.T * (wt * x)) * (x.T * (wt * y))
    return w

def funct(x1, x, tau):
    diff = x1 - x
    sq_diff = np.multiply(diff, diff)
    distances = np.sum(sq_diff, axis=1)
    weights_diag = np.exp(distances / (-2.0 * tau**2))
    return np.diag(np.ravel(weights_diag))

yMat = np.matrix(y)
tua = 0.9
yPred = lwr(xCon, yMat.T, tua)
plt.scatter(x, y)
plt.plot(x, yPred)
plt.show()


    
    

