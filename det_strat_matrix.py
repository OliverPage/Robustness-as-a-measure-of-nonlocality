#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:09:23 2022

@author: oliverpage

Generalise matrix of deterministic strategies
"""

import numpy as np
import itertools


no_dimensions = 3
no_measurements = 2

d2 = no_dimensions**2
n2 = no_measurements**2

def get_all_diag_values():
    """
    Get number of permuatations of a0_b0, a1_b1, ..., an_bn
    gives the number of deterministic strategies
    """
    arr = np.arange(0, d2, 1)
    diagonal_values_list = [p for p in itertools.product(arr, repeat=no_measurements)]
    return diagonal_values_list


def get_off_diag_values(diagonal):
    matrix = np.diag(diagonal)
    for i in range(no_measurements):
        bi_value = diagonal[i] % no_dimensions

        for o in range(no_dimensions):
            if diagonal[i] < (o + 1) * no_dimensions:
                ai_value = o
                break
            #else:
            #    a_value = no_dimensions - 1
        # Have ai_value (subsection) & bj_value (pos. in subsec.)
        for j in range(no_measurements):
            if i==j:
                break

            else:
                bj_value = diagonal[j] % no_dimensions
                for o in range(no_dimensions):
                    if diagonal[j] < (o + 1) * no_dimensions:
                        aj_value = o
                        break

            matrix[i, j] = no_dimensions * ai_value + bj_value
            matrix[j, i] = no_dimensions * aj_value + bi_value
    
    return matrix


def convert_mat_to_vec(mat):

    vec = np.zeros(n2*d2)
    count = 0
    for i in range(no_measurements):
        for k in range(no_measurements):
            
            vec[count*d2 + mat[i,k]] = 1
            count += 1
    return vec


def matrix_of_det_strats(): # any dimension and number of measurements

    no_rows = n2*d2
    no_cols = (no_dimensions**no_measurements)**2
    matrix =  np.zeros((no_rows, no_cols))
    diag_values = get_all_diag_values()
    count = 0 

    for Tuple in diag_values:
        strat_mat = get_off_diag_values(Tuple)
        strat_vec = convert_mat_to_vec(strat_mat)
        matrix[:, count] = strat_vec
        count += 1
    
    return matrix