#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:21:50 2022

@author: oliverpage
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

# Parameters
try:
    no_dimensions = int(sys.argv[1])
    no_measurements = int(sys.argv[2])
    no_games = int(sys.argv[3])

except:
    no_dimensions = 2
    no_measurements = 4
    no_games = int(1e2)
    
    

def load_all(no_measurements):

    with open("../../Data/OptRvsNd{}/Mean robustness, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "rb") as f:
        mean_robustness = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/Conditional mean robustness, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "rb") as f:
        cond_mean_robustness = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/NLV, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "rb") as f:
        NLV = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/Mean robustness error, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "rb") as f:
        mean_robustness_err = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/Conditional mean robustness error, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "rb") as f:
        cond_mean_robustness_err = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/NLV error, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "rb") as f:
        NLV_err = np.load(f)
        f.close()

    return mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err


def round_to_1(x):
   return np.round(x, -int(np.floor(np.log10(abs(x)))))

mean_robs = []
conditional_means = []
mean_rob_err = []
cond_mean_err = []
NLVs = []
NLVs_err = []

N = [2,3,4,5]#,6]

for n in N:
    mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err = load_all(n)
    mean_robs.append(mean_robustness)
    conditional_means.append(cond_mean_robustness)
    if n==6:
        mean_rob_err.append(mean_robustness_err[0])
        cond_mean_err.append(cond_mean_robustness_err[0])
        NLVs_err.append(round_to_1(NLV_err[0]))
        NLVs.append(np.round(NLV[0], len(str(NLVs_err[0]))-2))
    else:
        mean_rob_err.append(mean_robustness_err)
        cond_mean_err.append(cond_mean_robustness_err)
        NLVs_err.append(round_to_1(NLV_err))
        NLVs.append(np.round(NLV, len(str(NLVs_err[0]))-2))


mean_robs = np.array(mean_robs)
mean_robs = np.reshape(mean_robs, len(mean_robs))
conditional_means = np.array(conditional_means)
mean_rob_err = np.array(mean_rob_err)
cond_mean_err = np.array(cond_mean_err)
NLVs = np.array(NLVs)
NLVs_err = np.array(NLVs_err)

#NLVs_err = round_to_1(NLVs_err)

def new_plot_for_N(xdata, y1, y2, y1_err, y2_err, 
    y_label, title=None, savename=None, log = False):

    # Remove the if-statement for new data
    # if ylabel == "NLV":
    #     y_err = 3* y_err * np.sqrt((state_samples-1)/(no_games-1))

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Measurements')
    ax.set_ylabel(y_label)
    if log == True:
        ax.semilogy(xdata, y1, 'o', color='tab:red', label = "Mean")#, linestyle = "None")
    else:
        ax.plot(xdata, y1, 'o', color='tab:red', label = "Mean")#, linestyle = "None")

    ax.errorbar(xdata, y1, y1_err, color =  'tab:red')#, linestyle = "None")

    # Do second y axis
    if log == True:
        ax.semilogy(xdata, y2, 'o', color ='tab:blue', label = "Cond mean")#, linestyle = "None")
    else:
        ax.plot(xdata, y2, 'o', color ='tab:blue', label = "Cond mean")#, linestyle = "None")

    ax.errorbar(xdata, y2, y2_err, color = 'tab:blue')#, linestyle = "None")

    # Combine legend data
    #h1, l1 = ax1.get_legend_handles_labels()
    #h2, l2 = ax2.get_legend_handles_labels()
    #ax.legend(h1+h2, l1+l2, loc=2)
    ax.legend()
    ax.grid()
    
    i = 0
    for xy in zip(xdata, y1):
        if i>=len(xdata)-1:
            ax.annotate("NLV="+str(NLVs[i])+'±'+str(NLVs_err[i]), xy=[xy[0]*0.85, xy[1]*1.04], textcoords='data')
        else:
            ax.annotate("NLV="+str(NLVs[i])+'±'+str(NLVs_err[i]), xy=[xy[0]*1.07, xy[1]*1.02], textcoords='data')
        i += 1
    
    

    if title!=None:
        plt.title(title)

    if savename!=None:
        plt.savefig(savename+'.pdf')

    plt.show()
    

new_plot_for_N(N, mean_robs, conditional_means, mean_rob_err, cond_mean_err, 
    "Mean Robustness", None, savename="Robustness vs number of measurements, d={}".format(no_dimensions))
