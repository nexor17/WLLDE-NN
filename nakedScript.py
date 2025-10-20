import math
import os
from re import X
import numpy as np
import matplotlib.pyplot as plt
# Load open, high, low, and close prices from the CSV
input_csv = 'data/GSPC.csv'
open_prices, high_prices, low_prices, y = np.loadtxt(
    input_csv, delimiter=',', unpack=True, skiprows=1, usecols=(1, 2, 3, 4)
)
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

# Combine all data columns for the output CSV
output_data = np.vstack([x, open_prices, high_prices, low_prices, y, yPred]).T

# Define output directory and filename
output_dir = '/optidata'
base_filename = os.path.splitext(os.path.basename(input_csv))[0]
output_filename = f'{base_filename}Optimized.csv'
output_path = os.path.join(output_dir, output_filename)

# Create directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the combined data to the new CSV file
header = 'dayoftheyear,open,high,low,close,WLLDE'
np.savetxt(output_path, output_data, delimiter=',', header=header, comments='')

#plt.scatter(x, y)
#plt.plot(x, yPred)
#plt.show()


    
    

