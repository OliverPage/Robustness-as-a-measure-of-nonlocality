#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:48:04 2022

@author: oliverpage
"""
# Imports and required parameters

import numpy as np
import cvxpy as cp
import qutip as qp
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
import datetime
import itertools

# Parameters
no_dimensions = 4
no_measurements = 2

no_measures = int(1e2)
no_batches = 1

# Conversions
d2 = no_dimensions**2
n2 = no_measurements**2







#%% # Find anomalous state with paper method:

# Define parameters for the FT
alpha_0 = 0
alpha_1 = 1/2
beta_0 = 1/4
beta_1 = -1/4

phi_A0 = (2 * np.pi/no_dimensions) * alpha_0
phi_A1 = (2 * np.pi/no_dimensions) * alpha_1
phi_B0 = (2 * np.pi/no_dimensions) * beta_0 
phi_B1 = (2 * np.pi/no_dimensions) * beta_1 


def D_operator():

    D_A0 = 0
    D_A1 = 0
    D_B0 = 0
    D_B1 = 0

    for k in range(no_dimensions):
        k_state = qp.basis(no_dimensions, k)
        k_proj = k_state * k_state.dag()
        D_A0 +=  (1/np.sqrt(no_dimensions)) * np.exp(1j * phi_A0 * k) * k_proj
        D_A1 +=  (1/np.sqrt(no_dimensions)) * np.exp(-1j * phi_A1 * k) * k_proj
        D_B0 +=  (1/np.sqrt(no_dimensions)) * np.exp(1j * phi_B0 * k) * k_proj
        D_B1 +=  (1/np.sqrt(no_dimensions)) * np.exp(1j * phi_B1 * k) * k_proj

    return D_A0, D_A1, D_B0, D_B1 


def fourier_transform(obj):
    obj = obj.full()
    obj = np.fft.fftn(obj)
    obj = qp.Qobj(obj)
    return obj


def ifourier_transform(obj):
    obj = obj.full()
    obj = np.fft.ifftn(obj)
    obj = qp.Qobj(obj)
    return obj


def chi():

    D_A0, D_A1, D_B0, D_B1 = D_operator()

    chis_A0 = []
    chis_A1 = []
    chis_B0 = []
    chis_B1 = []

    for n in range(no_dimensions):
        #chis_A0.append((1/no_dimensions) * (qp.basis(no_dimensions,n).dag() * fourier_transform(D_A0)).dag())
        #chis_A1.append((1/no_dimensions) * (qp.basis(no_dimensions,n).dag() * fourier_transform(D_A1)).dag())
        #chis_B0.append((1/no_dimensions) * (qp.basis(no_dimensions,n).dag() * fourier_transform(D_B0)).dag())
        #chis_B1.append((1/no_dimensions) * (qp.basis(no_dimensions,n).dag() * fourier_transform(D_B1)).dag())

        chis_A0.append(D_A0.dag() * ifourier_transform(qp.basis(no_dimensions,n))*no_dimensions)
        chis_A1.append(D_A1.dag() * ifourier_transform(qp.basis(no_dimensions,n))*no_dimensions)
        chis_B0.append(D_B0.dag() * fourier_transform(qp.basis(no_dimensions,n)))
        chis_B1.append(D_B1.dag() * fourier_transform(qp.basis(no_dimensions,n)))

    return chis_A0, chis_A1, chis_B0, chis_B1


def measurement_tensor_product(a,x,b,y, 
                chis_A0, chis_A1, chis_B0, chis_B1):
    """
    calculate M_(a|x) x M_(b|y)
    """

    if x == 0:
        if y == 0:
            M_tensor = qp.tensor(chis_A0[a] * chis_A0[a].dag(), chis_B0[b] * chis_B0[b].dag())
        elif y == 1:
            M_tensor = qp.tensor(chis_A0[a] * chis_A0[a].dag(), chis_B1[b] * chis_B1[b].dag())

    elif x == 1:
        if y == 0:
            M_tensor = qp.tensor(chis_A1[a] * chis_A1[a].dag(), chis_B0[b] * chis_B0[b].dag())
        elif y == 1:
            M_tensor = qp.tensor(chis_A1[a] * chis_A1[a].dag(), chis_B1[b] * chis_B1[b].dag())
        
    return M_tensor


def sum_of_measurement_tensor_products(x, y, delta):

    chis_A0, chis_A1, chis_B0, chis_B1 = chi()
    Bell_term = 0
    for i in range(no_dimensions):
        Bell_term += measurement_tensor_product((i+delta)%no_dimensions, x,
                i%no_dimensions, y, chis_A0, chis_A1, chis_B0, chis_B1)
                 
    return Bell_term


def Bell_operator():

    Bell_op = 0

    #for k in range(no_dimensions):#
    for k in range(int(np.floor(no_dimensions/2))):
        """
        Bell_op += (1-2*k/(no_dimensions-1))*((sum_of_measurement_tensor_products(0, 0, k)
        +sum_of_measurement_tensor_products(1, 0, -k-1)
        +sum_of_measurement_tensor_products(1, 1, k)
        +sum_of_measurement_tensor_products(0, 1, -k))
        -(sum_of_measurement_tensor_products(0, 0, -k-1)
        +sum_of_measurement_tensor_products(1, 0, k)
        +sum_of_measurement_tensor_products(1, 1, -k-1)
        +sum_of_measurement_tensor_products(0, 1, k+1)))
        """
        Bell_op += (1 - (2*k/(no_dimensions-1))) * ((sum_of_measurement_tensor_products(0, 0, k)
        +sum_of_measurement_tensor_products(0, 1, -k)
        +sum_of_measurement_tensor_products(1, 0, -k)
        +sum_of_measurement_tensor_products(1, 1, k+2))
        -(sum_of_measurement_tensor_products(0, 0, -k+1)
        +sum_of_measurement_tensor_products(0, 1, k+2)
        +sum_of_measurement_tensor_products(1, 0, k+2)
        +sum_of_measurement_tensor_products(1, 1, -k)))
        

    return Bell_op

def get_anom_state():

    bell = Bell_operator()
    #eigenvalues = bell.eigenenergies()
    eigenvalues = bell.eigenstates()[0]
    eigenvectors = bell.eigenstates()[1]
    print("Largest eigenvalue = ", max(eigenvalues), '\n')
    print("Anomalous state = ", eigenvectors[eigenvalues.argmax()], '\n')

    

get_anom_state()

def ME_state():
    
    state = 0
    for i in range(no_dimensions):
        state += (1/np.sqrt(no_dimensions))*qp.tensor(qp.basis(no_dimensions,i), qp.basis(no_dimensions,i))

    return state

def ME_on_Bell():
    ME = ME_state()
    Bell = Bell_operator()
    
    return ME.dag() * Bell * ME
    
print("ME score = ", ME_on_Bell())





