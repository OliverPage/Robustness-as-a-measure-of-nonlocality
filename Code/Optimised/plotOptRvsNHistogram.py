#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  18 15:03:26 2022

@author: oliverpage
"""
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter 
import numpy as np
from tqdm import tqdm
import numpy as np
import sys

# Parameters
try:
    no_dimensions = int(sys.argv[1])
    no_measurements = int(sys.argv[2])
    no_games = int(sys.argv[3])

except:
    no_dimensions = 2
    no_measurements = 2
    no_games = int(1e4)
    
    
def round_to_1(x):
   return np.round(x, -int(np.floor(np.log10(abs(x)))))

with open("../../Data/OptRvsNd{}/robustness, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, 
            no_games, no_measurements, 0), "rb") as f:
            robustnesses = np.load(f)
            f.close()

#print(robustnesses)
robustnesses = abs(robustnesses)
print("Mean robustness = ", np.mean(robustnesses))

threshold = 1e-8
nonzero_robustness = robustnesses[abs(robustnesses)>threshold]
print("Conditional mean robustness =", np.mean(nonzero_robustness))

NLV = len(nonzero_robustness)/len(robustnesses)
print("NLV =", NLV)


#plot histogram
sns.set_style("darkgrid")
fig, axes = plt.subplots()
    
sns.histplot(data = nonzero_robustness, color = 'tab:purple', binwidth = 2*10**(-3))
axes.set_title("Optimised Robustness, {} games".format(no_games), size = 18)
axes.set_xlabel("Optimised Robustness", size = 18)
axes.set_ylabel("Frequency", size = 18)

plt.savefig("Optimised Robustness Histogram, d={}, m={}.pdf".format(no_dimensions, no_measurements))
plt.show()
