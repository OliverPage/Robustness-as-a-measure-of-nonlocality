#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 09:28:30 2022

@author: oliverpage
"""
# -*- coding: utf-8 -*-
# Imports and required parameters

import numpy as np
import cvxpy as cp
import qutip as qp
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
#import seaborn as sns
from tqdm import tqdm
import datetime
import itertools
import multiprocessing as mp
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
    



# Defining functions

def get_random_measurements():
    """
    Generate random projective measurements by taking the rows of unitary
    matrices generated using the Haar measure.

    Returns:
        measurements (numpy array): 6 projection matrices. The first three
        correspond to the input 0. The last three correspond to the input 1.
    """
    measurements = []

    for n in range(no_measurements):
        matrix = unitary_group.rvs(no_dimensions)
        for d in range(no_dimensions):
            vector = qp.Qobj(matrix[d])
            projector = vector * vector.dag()
            measurements.append(projector)

    return measurements


def sigma_ax(M_proj, state):
    """
    sigma_(a|x) = tr_A [(M_(a|x) tensor I)|state><state|]

    Parameters
    ----------
    M : qutip.qobj.Qobj
        Alice's measurement vector (matrix).
    state : qutip.qobj.Qobj
        the state vector (matrix).

    Returns
    -------
    qutip.qobj.Qobj
        sigma_(a|x) of the given set up.

    """
    # Get density operator
    state_projector = state * state.dag()
    # Construct sigma object
    #print(M_proj, qp.qeye(no_dimensions))
    M_tensor_I = qp.tensor(M_proj, qp.qeye(no_dimensions))
    thing_to_trace = M_tensor_I * state_projector

    return thing_to_trace.ptrace(1)


def probability(state, Alice_measurement, Bob_proj):
    """
    P(a,b|x,y) = <state|M_(a|x) tensor M_(b|y)|state>
                = tr[M_(a|x) tensor M_(b|y)|state><state|]
                = tr(sigma_(a|x)*M_(b|y)) #this is matrix multiplication not tensor product

    Parameters
    ----------
    state : qutip.qobj.Qobj
        state of the system (matrix).
    Alice_measurement : qutip.qobj.Qobj
        Alice's measurement vector (matrix).
    Bob_measurement : qutip.qobj.Qobj
        Bob's measurement vector (matrix).

    Returns
    -------
    real float
        probability of measuring a, b fiven x, y for measurements Axa & Byb.
    """
    #print(Alice_measurement, state)
    sigma = sigma_ax(Alice_measurement, state)
    if type(Bob_proj)==type(cp.Variable((4,4))):
        sigma_times_M = sigma @ Bob_proj
        print("\n\n", cp.real(cp.trace(sigma_times_M)), "\n\n")
        return cp.real(cp.trace(sigma_times_M))
        
    else:
        sigma_times_M = sigma * Bob_proj
        sigma_times_M = sigma_times_M.full()  # make it numpy rather than qutip
    
        return np.real(np.trace(sigma_times_M))


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


def test(mat):
    """
    1. Check if column elements sum to n2.
    
    2. Check if each segment has 1 one.
    """
    
    for j in range((no_dimensions**no_measurements)**2):
        vec = mat[:, j]
        if np.sum(vec) != n2:
            print("Error 1")
            return None
        
        for k in range(n2): # Loop over segments
            temp = 0
            for l in range(d2): # Loop over elements in the segments
                temp += vec[k*d2+l]
            if temp != 1:
                print("Error 2")
                return None

    print("All good!")
    return None


def state_vector(state, Alice_measurements, Bob_measurements):
    """ This is good for any dimension or number of measurements
    For two measurements per party only

    Parameters
    ----------
    state : qutip Qobj -> qp.qobj.Qobj
        Chosen state as a vector.
    Alice_measurements : qutip Qobj
        Array of measurement projectors for Alice.
        Alice_measurements = [A00, A01, A02, A10, A11, A12]
    Bob_measurements : qutip Qobj
        Array of measurement projectors for Bob.
        Bob_measurements = [B00, B01, B02, B10, B11, B12]

    Returns
    -------
    vector : numpy array
        The 36 dimensional probability vector with the same structure as in
        the columns of matrix_of_det_strats().
    """
    vector = np.zeros(n2*d2)
    count = 0
    vec_count = 0

    for a in range(no_measurements):
        for b in range(no_measurements):
            for i in range(a*no_dimensions, (a+1)*no_dimensions):
                for j in range(b * no_dimensions, (b + 1) * no_dimensions):
                    vector[count] = probability(state, Alice_measurements[i], Bob_measurements[j])
                    count += 1
            vector[(vec_count+1)*d2-1] = 1 - np.sum(vector[(vec_count)*d2:(vec_count+1)*d2-1])
            vec_count += 1

    return vector


def cvxpy_state_vector(state, Alice_measurements, Bob): # Unused
    """ This is good for any dimension or number of measurements
    For two measurements per party only

    Parameters
    ----------
    state : qutip Qobj -> qp.qobj.Qobj
        Chosen state as a vector.
    Alice_measurements : qutip Qobj
        Array of measurement projectors for Alice.
        Alice_measurements = [A00, A01, A02, A10, A11, A12]
    Bob_measurements : qutip Qobj
        Array of measurement projectors for Bob.
        Bob_measurements = [B00, B01, B02, B10, B11, B12]

    Returns
    -------
    vector : numpy array
        The 36 dimensional probability vector with the same structure as in
        the columns of matrix_of_det_strats().
    """
    vector = np.zeros(n2*d2)
    count = 0
    vec_count = 0
    
    Bob_measurements = []
    
    for row in range(no_measurements):
        for output in range(no_dimensions):
            Bob_measurements.append(Bob[row][output])

    for a in range(no_measurements):
        for b in range(no_measurements):
            for i in range(a*no_dimensions, (a+1)*no_dimensions):
                for j in range(b * no_dimensions, (b + 1) * no_dimensions):
                    temp = probability(state, Alice_measurements[i], Bob_measurements[j])
                    print('temp:', temp)
                    vector[count] = temp
                    count += 1
            vector[(vec_count+1)*d2-1] = 1 - np.sum(vector[(vec_count)*d2:(vec_count+1)*d2-1])
            vec_count += 1

    return vector


#----------------- Robustness OPtimisation for Bob  
# max_ent_state
def trace_for_sigma_by(A, state):
    A_tensor_Id = qp.tensor(A, qp.qeye(no_dimensions))
    state_proj = state*state.dag()
    thing_to_trace = A_tensor_Id * state_proj
    return thing_to_trace.ptrace(1)  # partial trace over Alice, same as sigma_ax?


def sigma_by(Bell, A, state, b, y):
    sigma = 0
    for a in range(no_measurements):
        for x in range(no_dimensions):
            # x gives which block of d^2
            # y gives position in block of d^2
            # a is multiple of m blocks
            # b is block in multiple of m blocks
            #print(f"a={a}, x={x}, b={b}, y={y}")
            sigma += Bell[(a*no_measurements + b)*d2 
                          + x*no_dimensions + y] * trace_for_sigma_by(A[a][x], state) # input A_(a|x)
    return sigma


def Bob_objective(Bell, Alice, Bob, state):
    thing_to_trace = 0
    for b in range(no_measurements):
        for y in range(no_dimensions):
            thing_to_trace += sigma_by(Bell, Alice, state, b, y) @ Bob[b][y]  # inputs for sigma_by
    return cp.real(cp.trace(thing_to_trace))
    

def init_Bob_opt(state, Alice=None, Bob=None):
    """ Get Bell operator for fixed Alice, and optimise Bob
    
                  Redundant - do not use
    """
    #1) Get random measurements for Alice and Bob
    if Alice==None and Bob==None:
        print("Measurements not given")
        Alice = get_random_measurements()
        Bob = get_random_measurements()
    else:
        print("Measurements given")
    
    #2) Calculate robustness for random measurements , extract Bell inequality
    robustness, Bell = SDP_opt(max_ent_state, Alice, Bob, get_Bell=True)
    #print("Found Bell vector")
    #print(f"Bell = {Bell}")
    
    #3) Optimise Bob's measurements
    # Set up Bob's measurements as cxpy variables
    Bob = []
    for y in range(no_measurements):
        row = []
        for b in range(no_dimensions):
            row.append(cp.Variable((no_dimensions, no_dimensions), hermitian=True))
        Bob.append(row)
    # Bob's measurements are stroed in a matrix with the following structure:
        # [[B00, B01, B02, ..., B0d],
        # [B10, B11, B12, ..., B1d],
        # .../
        # [Bm0, Bm1, Bm2, ..., Bmd]]
        
    # Format Alice's measurments to be in same format as Bob's
    Alice_formatted = []
    for i in range(no_measurements):
        Alice_formatted.append(Alice[i*no_dimensions:(i+1)*no_dimensions])
    #print("Reformatted measurement arrays")
    
    # Configure optimisation 
    objective = cp.Maximize(Bob_objective(Bell, Alice_formatted, Bob, state))
    #print("Set objective")
    
    constraints = []
    for y in range(no_measurements):
        sum_const = 0
        for b in range(no_dimensions):
            constraints.append(Bob[y][b]>>0)
            sum_const += Bob[y][b]
        constraints.append(sum_const<<identity)
    #print("Set constraints")
        
    prob = cp.Problem(objective, constraints)
    #print("Set problem")
    prob.solve(solver=cp.MOSEK)
    #print("Solved problem")
    
    # Reformatt measurements to be 1D array
    Bob_reformatted = []
    for row in range(no_measurements):
        for i in range(no_dimensions):
            Bob_reformatted.append(qp.Qobj(Bob[row][i].value))
            
    #print(f"Robustness = {robustness}")
    #print(f"Bob = {Bob_reformatted}")
    #print(f"Bell = {Bell}")
    # get robustness and Bell inequality
    #r_new, Bell_new = calculate_robustness_given_Bell(state, Alice, Bob_reformatted, Bell) #SDP_opt(max_ent_state, Alice, Bob_reformatted, get_Bell=True)
    
    #print("\nUsing new Bob and Bell operators...\n")
    #r_new=robustness
    #Bell_new = Bell
    ## Recursion if convergence not met
    #if abs(r_new-robustness)>1e-6:
    #    Bob_opt(state, Alice, Bob_reformatted)
    #else:
    #    return Bob_reformatted, r_new, Bell_new
    return Bob_reformatted, robustness, Bell
    

def optimise_bob_measurements(Alice, state, Bell):
    # Set up Bob's measurements as cxpy variables
    Bob = []
    for y in range(no_measurements):
        row = []
        for b in range(no_dimensions):
            if b==no_dimensions-1:
                row.append(identity - cp.sum(row))
            else:
                row.append(cp.Variable((no_dimensions, no_dimensions), hermitian=True))
        Bob.append(row)
    # Bob's measurements are stroed in a matrix with the following structure:
        # [[B00, B01, B02, ..., B0d],
        # [B10, B11, B12, ..., B1d],
        # .../
        # [Bm0, Bm1, Bm2, ..., Bmd]]
        
    # Format Alice's measurments to be in same format as Bob's
    Alice_formatted = []
    for i in range(no_measurements):
        Alice_formatted.append(Alice[i*no_dimensions:(i+1)*no_dimensions])
    #print("Reformatted measurement arrays")
    
    # Configure optimisation 
    objective = cp.Maximize(Bob_objective(Bell, Alice_formatted, Bob, state))
    #state_vec = cvxpy_state_vector(state, Alice, Bob)
    #objective = cp.Maximize((Bell @ state_vec - 1)/2)
    #print("Set objective")
    
    constraints = []
    for y in range(no_measurements):
        sum_const = 0
        for b in range(no_dimensions):
            constraints.append(Bob[y][b]>>0)
            sum_const += Bob[y][b]
        
        #constraints.append(sum_const<<identity)
        
    #print(constraints)
    #print("Set constraints")
        
    prob = cp.Problem(objective, constraints)
    #print("Set problem")
    r = prob.solve(solver=cp.MOSEK)
    #print("Solved problem")
    
    # Reformatt measurements to be 1D array
    Bob_reformatted = []
    for row in range(no_measurements):
        for i in range(no_dimensions):
            Bob_reformatted.append(qp.Qobj(Bob[row][i].value))
    
    return r, Bell, Bob_reformatted
 

def optimise_robustness(state, Alice=None, Bob=None, return_all=False):
    
    #1) Get random measurements for Alice and Bob
    if Alice==None:
        Alice = get_random_measurements()
    if Bob==None:
        Bob = get_random_measurements()
    
    #2) Calculate robustness given Alice and Bob, extract Bell operator
    robustness, Bell = SDP_opt(max_ent_state, Alice, Bob, get_Bell=True)
    print("First robustness:", robustness)
    #print("Get first Bell vector")
    #print(robustness, Bell)
    #old_robustness = 999
    robustness_list = [999, robustness]

    while abs(robustness_list[-2]-robustness_list[-1])>1e-8:  #abs(old_robustness-robustness)>1e-8: #abs(temp-robustness)>1e-8:
        #old_robustness = robustness
        #3) Optimise for Bob
        r, temp_Bell, Bob = optimise_bob_measurements(Alice, state, Bell)
        
        #4) Using new Bob, get new Bell, and new robustness
        robustness, Bell = SDP_opt(state, Alice, Bob, True)
        #print(count, new_robustness, Bell)
    
        #temp = robustness
        print("Robustness:", robustness)
        #print("Old Robustness:", old_robustness)
        robustness_list.append(robustness)
    
    print(robustness_list)
    
    if return_all:
        return robustness, Bell, Bob
    
    return robustness#, Bell, Bob


def test_optimised_robustness():
    Alice = get_random_measurements()
    Bob = get_random_measurements()
    

    rob_array = []
    Bob_array = []
    Bell_array = []
    
    for i in range(10):
        b = get_random_measurements()
        
        rob, Bell, new_Bob = optimise_robustness(max_ent_state, Alice, b, return_all=True)
        
        rob_array.append(rob)
        Bob_array.append(new_Bob)
        Bell_array.append(Bell)
    
    for i in rob_array:
        if not np.isclose(i, max(rob_array)):
            print("ERROR")
            print(max(rob_array), i)
        
    return rob_array, Bob_array, Bell_array



def SDP_opt(state, Alice_measurements, Bob_measurements, get_Bell=False):
    """
    Find the random robustness for the given state and measurements. Minimise
    epsilon such that (1-epsilon)*state36 + epsilon*max_mixed_state can be
    expressed as a convex combination of deterministic local strategies.

    Parameters
    ----------
    state : qutip.qobj.Qobj
        state of the system (matrix).
    Alice_measurement : qutip.qobj.Qobj
        Alice's measurement vector (matrix).
    Bob_measurement : qutip.qobj.Qobj
        Bob's measurement vector (matrix).

    Returns
    -------
    numpy float
        Epsilon, random robustness. Epsilon values 0<=epsilon<=1

    """
    # Problem data.
    # Make random measurement projectors that we will optimise
    # Use nonneg=True when initialising variable rather then in constraints
    # of SDP.
    #M = Model()

    I = cp.Variable(d2*n2)#(no_dimensions**no_measurements)**2))  # q_vec[i] >= 0, sum(q_vec[i]) = 1

    # Convert state from computational basis to 36 probability vector
    state_vec = state_vector(state, Alice_measurements, Bob_measurements)

    # Construct the problem.
    objective = cp.Maximize((I @ state_vec - 1)/2)

    # Set the remaining constraints
    constraints = [np.transpose(D) @ I >= -1, np.transpose(D) @ I <= 1]

    prob = cp.Problem(objective, constraints)
    
    r = prob.solve(solver=cp.MOSEK)
    
    if get_Bell:
        return r, I.value
    
    return r


def write_stats(filename, robustness_ME):
    """
    Parameters
    ----------
    filename : string
        file name to save the statistics of the datasets.

    Returns
    -------
    None.

    """
    nonzero_robustness_ME = np.delete(robustness_ME, np.where(robustness_ME < 1e-8))
    volume_ME = len(nonzero_robustness_ME) / len(robustness_ME)
    
    with open(filename, "w") as f:
        f.write("Number of dimensions = {}, number of measurements per party = {}\n".format(no_dimensions, no_measurements))
        f.write("Number of games = {}\n\n".format(no_games))
        f.write("Max entangled state:\n\n")
        f.write("Volume of nonlocality = {}\n\n".format(volume_ME))
        f.write("Total dataset:\n")
        f.write("Lowest score = {}\n".format(min(robustness_ME)))
        f.write("Highest score = {}\n".format(max(robustness_ME)))
        f.write("Mean score = {} ± {}\n".format(np.mean(robustness_ME),
                                                3*np.mean(robustness_ME) / np.sqrt(len(robustness_ME) - 1)))
        f.write("Median score = {}\n".format(np.median(robustness_ME)))
        f.write("STD of scores = {}\n".format(np.std(robustness_ME)))
        # f.write("Variance of scores = {}\n\n".format(np.var(robustness_ME)))

        f.write("Conditional dataset (remove zero r (epsilons)):\n")
        f.write("Lowest score = {}\n".format(min(nonzero_robustness_ME)))
        f.write("Mean score = {} ± {}\n".format(np.mean(nonzero_robustness_ME),
                                                3*np.mean(nonzero_robustness_ME) / np.sqrt(
                                                    len(nonzero_robustness_ME) - 1)))
        f.write("Median score = {}\n".format(np.median(nonzero_robustness_ME)))
        f.write("STD of scores = {}\n".format(np.std(nonzero_robustness_ME)))
        # f.write("Variance of scores = {}\n\n".format(np.var(nonzero_robustness_ME)))

        f.write("\n\nfile created at {}".format(datetime.datetime.now()))
        f.write("Showing three standard deviations for the uncertainty")
        f.close()


def plot(entanglement, y, y_err, ylabel, title):
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # Remove the if-statement for new data
    #if ylabel == "NLV":
    #    y_err = 3* y_err * np.sqrt((state_samples-1)/(no_games-1))
    
    ax.plot(entanglement, y, 'o', color = 'tab:purple')
    ax.errorbar(entanglement, y, yerr=y_err, fmt='o')
    ax.set_xlabel("Entanglement", size = 16)
    ax.set_ylabel(ylabel, size = 16)
    
    plt.savefig(title + ".pdf")
    plt.show()


def plot_two_axis(xdata, y1, y2, y1_err, y2_err, 
    y_label, y2_label, title=None, savename=None, log = False):

    # Remove the if-statement for new data
    # if ylabel == "NLV":
    #     y_err = 3* y_err * np.sqrt((state_samples-1)/(no_games-1))

    # Order
    entanglement_index = np.argsort(xdata)
    xdata = [xdata[i] for i in entanglement_index]
    y1 = [y1[i] for i in entanglement_index]
    y2 = [y2[i] for i in entanglement_index]
    y1_err = [y1_err[i] for i in entanglement_index]
    y2_err = [y2_err[i] for i in entanglement_index]

    # Get rid of zero elements to avoid taking the log of 0
    y1 = np.delete(y1, np.where(y1==0))
    y2 = np.delete(y2, np.where(y2==0))
    length = min(len(y1), len(y2))

    y1 = y1[-length::]
    y2 = y2[-length::]

    xdata = xdata[-length::]
    y1_err = y1_err[-length::]
    y2_err = y2_err[-length::]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Entanglement')
    ax1.set_ylabel(y_label)
    if log == True:
        ax1.semilogy(xdata, y1, 'o', color='tab:red', label = "Mean", linestyle = "None")
    else:
        ax1.plot(xdata, y1, 'o', color='tab:red', label = "Mean", linestyle = "None")

    ax1.errorbar(xdata, y1, y1_err, color =  'tab:red', linestyle = "None")

    # Do second y axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_label)
    if log == True:
        ax2.semilogy(xdata, y2, 'o', color ='tab:blue', label = "Cond mean", linestyle = "None")
    else:
        ax2.plot(xdata, y2, 'o', color ='tab:blue', label = "Cond mean", linestyle = "None")

    ax2.errorbar(xdata, y2, y2_err, color = 'tab:blue', linestyle = "None")

    # Combine legend data
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=2)

    if title!=None:
        plt.title(title)

    if savename!=None:
        plt.savefig(savename+'.pdf')

    plt.show()


def calculate(no_games, seed):
    """
    Calculate random robustness for no_games.

    Parameters
    ----------
    no_games : int
        number of random games to run.

    Returns
    -------
    max_entangled_epsilons : numpy array (floats)
        Random robustness epsilons for no_games and max entangled state.

    """

    robustness = np.zeros(no_games)
    for i in tqdm(range(no_games), desc = "Generating robustness..."):
        # Get random projective measurements
        #A_measurements = get_random_measurements()
        #B_measurements = get_random_measurements()
        
        # Get max entangled robustness
        robustness[i] = optimise_robustness(max_ent_state) #SDP_opt(max_ent_state, A_measurements, B_measurements)

    return robustness


def random_numbers(no):
    rand_nos = np.random.randint(0, int(2 ** 32 - 1), dtype = np.int64, size=no)
    if len(np.unique(rand_nos)) != no:
        random_numbers(no)

    return rand_nos


def save(mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err, batch_no):
    
    with open("../../Data/OptRvsNd{}/Mean robustness, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "wb") as f:
        np.save(f, mean_robustness)
        f.close()
            
    with open("../../Data/OptRvsNd{}/Conditional mean robustness, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "wb") as f:
        np.save(f, cond_mean_robustness)
        f.close()
        
    with open("../../Data/OptRvsNd{}/NLV, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "wb") as f:
        np.save(f, NLV)
        f.close()
        
    with open("../../Data/OptRvsNd{}/Mean robustness error, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "wb") as f:
        np.save(f, mean_robustness_err)
        f.close()
            
    with open("../../Data/OptRvsNd{}/Conditional mean robustness error, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "wb") as f:
        np.save(f, cond_mean_robustness_err)
        f.close()
        
    with open("../../Data/OptRvsNd{}/NLV error, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "wb") as f:
        np.save(f, NLV_err)
        f.close()
        
        
def save_all(mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err):
    
    with open("../../Data/OptRvsNd{}/Mean robustness, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "wb") as f:
        np.save(f, mean_robustness)
        f.close()
            
    with open("../../Data/OptRvsNd{}/Conditional mean robustness, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "wb") as f:
        np.save(f, cond_mean_robustness)
        f.close()
        
    with open("../../Data/OptRvsNd{}/NLV, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "wb") as f:
        np.save(f, NLV)
        f.close()
        
    with open("../../Data/OptRvsNd{}/Mean robustness error, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "wb") as f:
        np.save(f, mean_robustness_err)
        f.close()
            
    with open("../../Data/OptRvsNd{}/Conditional mean robustness error, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "wb") as f:
        np.save(f, cond_mean_robustness_err)
        f.close()
        
    with open("../../Data/OptRvsNd{}/NLV error, no_games {}, no_meas {}, all.npy".format(no_dimensions, no_games, no_measurements), "wb") as f:
        np.save(f, NLV_err)
        f.close()


def load(no_measurements, batch_no):

    with open("../../Data/OptRvsNd{}/Mean robustness, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "rb") as f:
        mean_robustness = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/Conditional mean robustness, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "rb") as f:
        cond_mean_robustness = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/NLV, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "rb") as f:
        NLV = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/Mean robustness error, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "rb") as f:
        mean_robustness_err = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/Conditional mean robustness error, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "rb") as f:
        cond_mean_robustness_err = np.load(f)
        f.close()
        
    with open("../../Data/OptRvsNd{}/NLV error, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, no_games, no_measurements, batch_no), "rb") as f:
        NLV_err = np.load(f)
        f.close()

    return mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err


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


def parallel_data_production(batch_no=0):

    with mp.Pool() as pool:
        results = pool.starmap(calculate,
                               inputs[no_cores * batch_no:no_cores * (batch_no + 1)])  # [no_games_per_core]*no_cores)

    # Recast into 1D array of robustness values
    results_array = np.array([])
    for i in range(no_cores):
        results_array = np.concatenate((results_array, results[i]))
    # Save full dataset of robustness values
    with open("../../Data/OptRvsNd{}/robustness, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, 
                no_games, no_measurements, batch_no), "wb") as f:
                np.save(f, results_array)
                f.close()


    #robustness = np.concatenate([results[n] for n in range(no_cores)])

    #mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err = analyse(robustness)
    #save(mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err, batch_no)

    # write_stats("Robustness, no games {}, batch number {}.txt".format(no_games, batch_no), batch_no)


def analyse(robustness):
    
    robustness[robustness<1e-10] = 0
    
    nonzero_robustness = np.delete(robustness, np.where(robustness < 1e-8))

    mean_robustness = np.mean(robustness)
    NLV = len(nonzero_robustness) / len(robustness)

    mean_robustness_err = 3*np.mean(robustness) / np.sqrt(len(robustness) - 1)
    
    if len(nonzero_robustness) == 0 or len(nonzero_robustness) == 1:
        cond_mean_robustness = 0
        cond_mean_robustness_err = 0
        NLV_err = 0
        
    else: 
        cond_mean_robustness = np.mean(nonzero_robustness)
        cond_mean_robustness_err = 3*np.mean(nonzero_robustness) / np.sqrt(len(nonzero_robustness) - 1)
        NLV_err = 3*np.sqrt((NLV-NLV**2)/(len(nonzero_robustness)-1))
    
    return mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err


def order(entanglement, ydata):

    entanglement_index = np.argsort(entanglement)
    entanglement = [entanglement[i] for i in entanglement_index]
    ydata = [ydata[i] for i in entanglement_index]

    return entanglement, ydata


def ME_state():
    
    state = 0
    for i in range(no_dimensions):
        state += (1/np.sqrt(no_dimensions))*qp.tensor(qp.basis(no_dimensions,i), qp.basis(no_dimensions,i))

    return state  


# Parameters
#no_dimensions = 2
#no_measurements = 2

#no_games = 10**4
no_batches = 1
no_games_per_batch = int(no_games / no_batches)

# Conversions
d2 = no_dimensions**2
n2 = no_measurements**2

max_ent_state = ME_state()
D = matrix_of_det_strats()
identity = np.identity(no_dimensions)

robustness = np.array([])
entanglement_all = np.array([])
mean_robustness_all = np.array([])
cond_mean_robustness_all = np.array([])
NLV_all = np.array([])
mean_robustness_err_all = np.array([])
cond_mean_robustness_err_all = np.array([])
NLV_err_all = np.array([])

mode = 'test'

if mode=='test':
    #r, bell, bob = optimise_robustness(max_ent_state)
    r = optimise_robustness(max_ent_state)
    print("\nEnd:")
    print(f"Robustness = {r}")
    #print(f"Bob = {bob}")
    #print(f"Bell = {bell}")
    """
    Bob, rob, Bell = Bob_opt(max_ent_state)
    print("\nEnd:")
    print(f"Robustness = {rob}")
    print(f"Bob = {Bob}")
    print(f"Bell = {Bell}")
    """

elif mode == 'calculate':
    
    if __name__ == '__main__':
        no_cores = mp.cpu_count() - 1
        no_games_per_core = int(no_games_per_batch/no_cores)
        random_seeds = random_numbers(no_cores * no_batches)
        inputs = [(no_games_per_core, seed) for seed in random_seeds]
        print("Number fo dimensions = {}".format(no_dimensions))
        print("Number of measurements =", no_measurements)
        print("Number of unique random seeds = " + str(len(list(zip(*inputs))[1])))
        print("Number of cores = " + str(no_cores))
        print("Number of batches = " + str(no_batches) + "\n")

        for i in range(no_batches):
            print("Batch number = " + str(i))
            parallel_data_production(i)
    
        for batch_no in range(no_batches):
            with open("../../Data/OptRvsNd{}/robustness, no_games {}, no_meas {}, batch {}.npy".format(no_dimensions, 
                no_games, no_measurements, batch_no), "rb") as f:
                R = np.load(f)
                f.close()
            robustness = np.concatenate((robustness, R))

        mean_robustness_all, cond_mean_robustness_all, NLV_all, mean_robustness_err_all, cond_mean_robustness_err_all, NLV_err_all = analyse(robustness)
        #save(mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err, batch_no)

        """
        mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err = load(no_measurements, batch_no)
        mean_robustness_all = np.append(mean_robustness_all, mean_robustness)
        cond_mean_robustness_all = np.append(cond_mean_robustness_all, cond_mean_robustness)
        NLV_all = np.append(NLV_all, NLV)
        mean_robustness_err_all = np.append(mean_robustness_err_all, mean_robustness_err)
        cond_mean_robustness_err_all = np.append(cond_mean_robustness_err_all, cond_mean_robustness_err)
        NLV_err_all = np.append(NLV_err_all, NLV_err)
        """
        save_all(mean_robustness_all, cond_mean_robustness_all, NLV_all, mean_robustness_err_all, cond_mean_robustness_err_all, NLV_err_all)
        print('saved all')
        #plot(NLV_all, NLV_err_all, "NLV", "NLV vs Entanglement")
        #plot_two_axis(entanglement_all, mean_robustness_all, cond_mean_robustness_all, mean_robustness_err_all, cond_mean_robustness_err_all, 
        #    "Mean robustness", "Conditional mean robustness", title="Robustness vs Entanglement", savename="Robustness vs Entanglement", log = True)
        #plot_two_axis(entanglement_all, mean_robustness_all, cond_mean_robustness_all, mean_robustness_err_all, cond_mean_robustness_err_all, 
        #    "Mean robustness", "Conditional mean robustness", title="Robustness vs Entanglement", savename="Robustness vs Entanglement", log = False)


elif mode == 'load':
    mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err = load_all(no_measurements)

    #plot(entanglement, NLV, NLV_err, "NLV", "NLV vs Entanglement")
    #plot_two_axis(entanglement, mean_robustness, cond_mean_robustness, mean_robustness_err, cond_mean_robustness_err, 
    #    "Mean robustness", "Conditional mean robustness", title="Robustness vs Entanglement", savename="Robustness vs Entanglement", log = True)
    #plot_two_axis(entanglement, mean_robustness, cond_mean_robustness, mean_robustness_err, cond_mean_robustness_err, 
    #    "Mean robustness", "Conditional mean robustness", title="Robustness vs Entanglement", savename="Robustness vs Entanglement", log = False)
    
#%% Plotting

    