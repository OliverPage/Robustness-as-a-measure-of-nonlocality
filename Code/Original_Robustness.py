#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:59:23 2022

@author: oliverpage
"""

import numpy as np
import cvxpy as cp
import qutip as qp
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
import datetime


def get_random_measurements():
    """
    Generate random projective measurements by taking the rows of unitary 
    matrices generated using the Haar measure.
    
    Returns:
        measurements (numpy array): 6 projection matrices. The first three 
        correspond to the input 0. The last three correspond to the input 1.
    """

    A0 = unitary_group.rvs(3)
    A1 = unitary_group.rvs(3)

    A00 = qp.Qobj(A0[0])
    A01 = qp.Qobj(A0[1])
    A02 = qp.Qobj(A0[2])

    A10 = qp.Qobj(A1[0])
    A11 = qp.Qobj(A1[1])
    A12 = qp.Qobj(A1[2])

    A00 = A00 * A00.dag()
    A01 = A01 * A01.dag()
    A02 = A02 * A02.dag()

    A10 = A10 * A10.dag()
    A11 = A11 * A11.dag()
    A12 = A12 * A12.dag()

    return [A00,A01,A02,A10,A11,A12]


def get_paper_measurements():
    """
    Define paper measurements from https://arxiv.org/abs/quant-ph/0601210v1.
    
    Returns:
        A_measurements, B_measurements (lists): Alice's and Bob's projection
        matrix measurements, 6 each.
    """
    # Player-input-outcome; e.g. A12 is Alice measurement 1 outcome 2
    # Get measurement vectors
    A00 = qp.Qobj((1/np.sqrt(3))*np.array([1, np.exp(alpha_0*1j), np.exp(alpha_0*2j)]))
    A01 = qp.Qobj((1/np.sqrt(3))*np.array([1, chi*np.exp(alpha_0*1j), np.conj(chi)*np.exp(alpha_0*2j)]))
    A02 = qp.Qobj((1/np.sqrt(3))*np.array([1, np.conj(chi)*np.exp(alpha_0*1j), chi*np.exp(alpha_0*2j)]))
    A10 = qp.Qobj((1/np.sqrt(3))*np.array([1, np.exp(alpha_1*1j), np.exp(alpha_1*2j)]))
    A11 = qp.Qobj((1/np.sqrt(3))*np.array([1, chi*np.exp(alpha_1*1j), np.conj(chi)*np.exp(alpha_1*2j)]))
    A12 = qp.Qobj((1/np.sqrt(3))*np.array([1, np.conj(chi)*np.exp(alpha_1*1j), chi*np.exp(alpha_1*2j)]))
    
    # Turn into projection matrices
    A00 = A00 * A00.dag()
    A01 = A01 * A01.dag()
    A02 = A02 * A02.dag()
    A10 = A10 * A10.dag()
    A11 = A11 * A11.dag()
    A12 = A12 * A12.dag()
    
    # Repeat for Bob
    B00 = qp.Qobj((1/np.sqrt(3))*np.array([1, np.exp(beta_0*1j), np.exp(beta_0*2j)]))
    B01 = qp.Qobj((1/np.sqrt(3))*np.array([1, np.conj(chi)*np.exp(beta_0*1j), chi*np.exp(beta_0*2j)]))
    B02 = qp.Qobj((1/np.sqrt(3))*np.array([1, chi*np.exp(beta_0*1j), np.conj(chi)*np.exp(beta_0*2j)]))
    B10 = qp.Qobj((1/np.sqrt(3))*np.array([1, np.exp(beta_1*1j), np.exp(beta_1*2j)]))
    B11 = qp.Qobj((1/np.sqrt(3))*np.array([1, np.conj(chi)*np.exp(beta_1*1j), chi*np.exp(beta_1*2j)]))
    B12 = qp.Qobj((1/np.sqrt(3))*np.array([1, chi*np.exp(beta_1*1j), np.conj(chi)*np.exp(beta_1*2j)]))
    
    B00 = B00 * B00.dag()
    B01 = B01 * B01.dag()
    B02 = B02 * B02.dag()
    B10 = B10 * B10.dag()
    B11 = B11 * B11.dag()
    B12 = B12 * B12.dag()
    
    A_measurements = [A00,A01,A02,A10,A11,A12]
    B_measurements = [B00,B01,B02,B10,B11,B12]

    return A_measurements, B_measurements


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
    M_tensor_I = qp.tensor(M_proj,qp.qeye(3))
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
    sigma = sigma_ax(Alice_measurement, state)
    sigma_times_M = sigma * Bob_proj
    sigma_times_M = sigma_times_M.full() # make it numpy rather than qutip
    
    return np.real(np.trace(sigma_times_M))


def matrix_of_det_strats():
    """
    Generate a 36x81 matrix with 0's and 1's where each column corresponding
    to a deterministic strategy. 36 comes from 36 possible combinations of
    (a, b, x, y).
    
    Structure of a det strat column is as follows. The first 9 are
    probabilities of getting a and b (values 0,1,2) given x=0, y=0, the next 9
    are for x=0, y=1, and then x=1, y=0, and finally x=1, y=1.
    Order of a and b is 00,01,02,10,11,12,20,21,22.
    
    a gives which triple, b gives position in the triple
    
    Returns
    -------
    matrix : numpy array
        matrix of deterministic strategies.

    """
    matrix = np.zeros((36, 81))
    count = 0
    
    for a0_b0 in range(9): # set first 9 entries P(a,b|00)
        vec1 = np.zeros(9)
        vec1[a0_b0] = 1
        
        if a0_b0<3:
            a0 = 0
            
        elif a0_b0<6:
            a0 = 1
            
        else:
            a0 = 2
        
        b0 = a0_b0%3
        
        for a1_b1 in range(9): # set last 9 entries P(a,b|11)
            vec4 = np.zeros(9)
            vec4[a1_b1] = 1

            if a1_b1<3:
                a1 = 0
                
            elif a1_b1<6:
                a1 = 1
                
            else:
                a1 = 2
            
            b1 = a1_b1%3
            
            a0_b1 = 3*a0+b1
            a1_b0 = 3*a1+b0
            
            # Find vec2 & vec3
            vec2 = np.zeros(9)
            vec2[a0_b1] = 1
            vec3 = np.zeros(9)
            vec3[a1_b0] = 1
            
            det_strat = np.concatenate((vec1, vec2, vec3, vec4))
            matrix[:,count] = det_strat
            count += 1
            
    return matrix


def state_36vector(state, Alice_measurements, Bob_measurements):
    """
    

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
    vector = np.zeros(36)
    count = 0
    for i in range(3): # inputs 00
        for j in range(3):
            vector[count] = probability(state, Alice_measurements[i], Bob_measurements[j])
            count += 1
    
    for i in range(3): # inputs 01
        for j in range(3,6):
            vector[count] = probability(state, Alice_measurements[i], Bob_measurements[j])
            count += 1
    
    for i in range(3, 6): # inputs 10
        for j in range(3):
            vector[count] = probability(state, Alice_measurements[i], Bob_measurements[j])
            count += 1
    
    for i in range(3, 6): # inputs 11
        for j in range(3,6):
            vector[count] = probability(state, Alice_measurements[i], Bob_measurements[j])
            count += 1
            
    # Normalisation to get the final values of each set of 9
    vector[8] = 1 - np.sum(vector[0:8])
    vector[17] = 1 - np.sum(vector[9:17])
    vector[26] = 1 - np.sum(vector[18:26])
    vector[35] = 1 - np.sum(vector[27:35])
    
    return vector


def SDP_opt(state, Alice_measurements, Bob_measurements):
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
    r = cp.Variable(nonneg=True)
    s = cp.Variable((81,), nonneg=True) # q_vec[i] >= 0, sum(q_vec[i]) = 1
    q = cp.Variable((81,), nonneg=True)

    # Convert state from computational basis to 36 probability vector
    state36 = state_36vector(state, Alice_measurements, Bob_measurements)
    
    # Construct the problem.     
    objective = cp.Minimize(r)
    # Set the remaining constraints
    constraints = [state36 + (deterministic_matrix@s) == (deterministic_matrix@q), 
                   cp.sum(q) == 1+r, cp.sum(s) == r] 
    prob = cp.Problem(objective, constraints)
    
    return prob.solve(solver=cp.MOSEK)


def write_stats(filename, robustness_ME, robustness_anom):
    """
    Parameters
    ----------
    filename : string
        file name to save the statistics of the datasets.

    Returns
    -------
    None.

    """
    nonzero_robustness_ME = np.delete(robustness_ME, np.where(robustness_ME<1e-8))
    nonzero_robustness_anom = np.delete(robustness_anom, np.where(robustness_anom<1e-8))

    volume_ME = len(nonzero_robustness_ME)/len(robustness_ME)
    volume_anom = len(nonzero_robustness_anom)/len(robustness_anom)
        
    with open(filename, "w") as f:
            f.write("Number of games = {}\n\n".format(no_measures))
            f.write("Max entangled state:\n\n")
            f.write("Volume of nonlocality = {}\n\n".format(volume_ME))
            f.write("Total dataset:\n")
            f.write("Lowest score = {}\n".format(min(robustness_ME)))
            f.write("Highest score = {}\n".format(max(robustness_ME)))
            f.write("Mean score = {} ± {}\n".format(np.mean(robustness_ME), np.mean(robustness_ME)/np.sqrt(len(robustness_ME)-1)))
            f.write("Median score = {}\n".format(np.median(robustness_ME)))
            f.write("STD of scores = {}\n".format(np.std(robustness_ME)))
            #f.write("Variance of scores = {}\n\n".format(np.var(robustness_ME)))
            
            f.write("Conditional dataset (remove zero r (epsilons)):\n")
            f.write("Lowest score = {}\n".format(min(nonzero_robustness_ME)))
            f.write("Mean score = {} ± {}\n".format(np.mean(nonzero_robustness_ME),np.mean(nonzero_robustness_ME)/np.sqrt(len(nonzero_robustness_ME)-1)))
            f.write("Median score = {}\n".format(np.median(nonzero_robustness_ME)))
            f.write("STD of scores = {}\n".format(np.std(nonzero_robustness_ME)))
            #f.write("Variance of scores = {}\n\n".format(np.var(nonzero_robustness_ME)))
            
            f.write("\nAnomalous state:\n\n")
            f.write("Volume of nonlocality = {}\n\n".format(volume_anom))
            f.write("Total dataset:\n")
            f.write("Lowest score = {}\n".format(min(robustness_anom)))
            f.write("Highest score = {}\n".format(max(robustness_anom)))
            f.write("Mean score = {} ± {}\n".format(np.mean(robustness_anom), np.mean(robustness_anom)/np.sqrt(len(robustness_anom)-1)))
            f.write("Median score = {}\n".format(np.median(robustness_anom)))
            f.write("STD of scores = {}\n".format(np.std(robustness_anom)))
            #f.write("Variance of scores = {}\n\n".format(np.var(robustness_anom)))
            
            f.write("Conditional dataset (remove zero r (epsilons)):\n")
            f.write("Lowest score = {}\n".format(min(nonzero_robustness_anom)))
            f.write("Mean score = {} ± {}\n".format(np.mean(nonzero_robustness_anom), np.mean(nonzero_robustness_anom)/np.sqrt(len(nonzero_robustness_anom)-1)))
            f.write("Median score = {}\n".format(np.median(nonzero_robustness_anom)))
            f.write("STD of scores = {}\n".format(np.std(nonzero_robustness_anom)))
            #f.write("Variance of scores = {}\n\n".format(np.var(nonzero_robustness_anom)))
            
            f.write("\n\nfile created at {}".format(datetime.datetime.now()))
            f.close()


def plot(max_entangled_state, anomalous_state, no_games, savename=None, leg_loc=1, inc_zeros=False):
    """
    

    Parameters
    ----------
    max_entangled_state : qutip.qobj.Qobj 
        max entangled state vector.
    anomalous_state : qutip.qobj.Qobj 
        anomalous state vector.
    no_games : int
        number of random games to run.
    savename : string, optional
        Name of the pdf to be saved + ", {no_games} Games, Both States.pdf"

    Returns
    -------
    None.

    """
    # Get conditionals
    nonzero_robustness_ME = np.delete(max_entangled_state, np.where(max_entangled_state<1e-8))
    nonzero_robustness_anom = np.delete(anomalous_state, np.where(anomalous_state<1e-8))
    zero_robustness_ME = np.delete(max_entangled_state, np.where(max_entangled_state>1e-8))
    zero_robustness_anom = np.delete(anomalous_state, np.where(anomalous_state>1e-8))
    
    # Get NLV
    NLV_ME = np.round(len(nonzero_robustness_ME)/len(max_entangled_state),4)
    NLV_anom = np.round(len(nonzero_robustness_anom)/len(anomalous_state),4)
    
    # Get means
    mean_ME = np.mean(max_entangled_state)
    mean_ME_err = mean_ME/np.sqrt(len(max_entangled_state)-1)
    
    conditional_mean_ME = np.mean(nonzero_robustness_ME)
    conditional_mean_ME_err = conditional_mean_ME/np.sqrt(len(nonzero_robustness_ME)-1)
    
    mean_anom = np.mean(anomalous_state)
    mean_anom_err = mean_anom/np.sqrt(len(anomalous_state)-1)
    
    conditional_mean_anom = np.mean(nonzero_robustness_anom)
    conditional_mean_anom_err = conditional_mean_anom/np.sqrt(len(nonzero_robustness_anom)-1)

    # Plot parameters
    no_bins = 40
    upper_limit = max(max(nonzero_robustness_ME), max(nonzero_robustness_anom))
    mean_height = len(nonzero_robustness_ME)/no_bins
    if inc_zeros:
        mean_height = len(max_entangled_state)/no_bins
    conditional_mean_height = mean_height
    
    #plot histogram
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(1, 2, figsize = (15,7))

    sns.histplot(data = nonzero_robustness_ME, color = 'tab:purple', ax = axes[0], 
                 bins=np.linspace(1e-8, upper_limit, no_bins))#,label="Nonzero scores")
    if inc_zeros:
        sns.histplot(data = zero_robustness_ME, color = 'tab:red', ax = axes[0], bins=[0, 1e-3],label="Zero scores, NLV = {}".format(NLV_ME))
    #axes[0].plot([mean_ME], [mean_height], 'k.', label='Mean = {}'.format(np.round(mean_ME,4)), markersize = 10)
    axes[0].errorbar([mean_ME], [mean_height], xerr=mean_ME_err, fmt = 'o', 
                     label='Mean = {}'.format(np.round(mean_ME,4)), markersize = 5)
    #axes[0].plot([conditional_mean_ME], [conditional_mean_height], 'r.', label='Conditional Mean = {}'.format(np.round(conditional_mean_ME,4)), markersize = 10)
    axes[0].errorbar([conditional_mean_ME], [conditional_mean_height], xerr=conditional_mean_ME_err, fmt = 'o', 
                     label='Conditional Mean = {}'.format(np.round(conditional_mean_ME,4)), markersize = 5)
    axes[0].set_title("Maximally Entangled State", size = 18)
    axes[0].set_xlabel(r"$\frac{\epsilon}{1-\epsilon}$", size = 18)
    axes[0].set_ylabel("Frequency", size = 18)
    axes[0].legend(loc=leg_loc)#prop={'size': 15}, loc=2)

    sns.histplot(data = nonzero_robustness_anom, color = 'tab:purple', ax = axes[1], 
                 bins=np.linspace(1e-8, upper_limit, no_bins))#,label="Nonzero scores")
    if inc_zeros:
        sns.histplot(data = zero_robustness_anom, color = 'tab:red', ax = axes[1], bins=[0, 1e-3],label="Zero scores, NLV = {}".format(NLV_anom))
    #axes[1].plot([mean_anom], [mean_height], 'k.', label='Mean = {}'.format(np.round(mean_anom,4)), markersize = 10)
    axes[1].errorbar([mean_anom], [mean_height], xerr=mean_anom_err, fmt = 'o',
                 label='Mean = {}'.format(np.round(mean_anom,4)), markersize = 5)
    #axes[1].plot([conditional_mean_anom], [conditional_mean_height], 'r.', label='Conditional Mean = {}'.format(np.round(conditional_mean_anom,4)), markersize = 10)
    axes[1].errorbar([conditional_mean_anom], [conditional_mean_height], xerr=conditional_mean_anom_err, fmt = 'o',
                 label='Conditional Mean = {}'.format(np.round(conditional_mean_anom,4)), markersize = 5)
    axes[1].set_title("Anomalous State", size = 18)
    axes[1].set_xlabel(r"$\frac{\epsilon}{1-\epsilon}$", size = 18)
    axes[1].set_ylabel("Frequency", size = 18)
    axes[1].legend(loc=leg_loc)#prop={'size': 15}, loc=2)

    if savename!=None:
        plt.savefig(savename+", {} Games, Both States.pdf".format(no_games), bbox_inches='tight')
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
    anomalous_epsilons : numpy array (floats)
        Random robustness epsilons for no_games and anomalous state.

    """
    max_entangled_epsilons = np.zeros(no_games)
    anomalous_epsilons = np.zeros(no_games)
    np.random.seed(seed)
    
    for i in tqdm(range(no_games), desc = "Running games..." ):
        # Get random projective measurements
        A_measurements = get_random_measurements()
        B_measurements = get_random_measurements()
        # Get max entangled robustness
        max_entangled_epsilon = SDP_opt(maxEntangledState, A_measurements, B_measurements)
        max_entangled_epsilons[i] = max_entangled_epsilon
        # Get anomalous robustness
        anomalous_epsilon = SDP_opt(anomalousState, A_measurements, B_measurements)
        anomalous_epsilons[i] = anomalous_epsilon
    
    return max_entangled_epsilons, anomalous_epsilons


def parallel_data_production(no_games, batch_no=0):
    
    with mp.Pool() as pool:
        results = pool.starmap(calculate, inputs[no_cores*batch_no:no_cores*(batch_no+1)])#[no_games_per_core]*no_cores)
        
    epsilon_max_entangled = np.concatenate([results[n][0] for n in range(no_cores)])
    epsilon_anomalous = np.concatenate([results[n][1] for n in range(no_cores)])
    
    with open("ME_out/max entangled epsilons, no games {}, batch {}.npy".format(no_games, batch_no), "wb") as f:
        np.save(f, epsilon_max_entangled)
        f.close()
    with open("anom_out/anomalous epsilons, no games {}, batch {}.npy".format(no_games, batch_no), "wb") as f:
        np.save(f, epsilon_anomalous)
        f.close()
    
    #write_stats("Robustness, no games {}, batch number {}.txt".format(no_games, batch_no), batch_no)


def random_numbers(no):
    rand_nos = np.random.randint(0, int(2**32 - 1), size=no)
    if len(np.unique(rand_nos))!=no:
        random_numbers(no)

    return rand_nos   
# ------------------------------- Define variables----------------------------

# Set up Qutip variables
zero = qp.basis(3,0)
one = qp.basis(3,1)
two = qp.basis(3,2)
zero_zero = qp.tensor(zero,zero)
one_one = qp.tensor(one,one)
two_two = qp.tensor(two,two)

# Variables to simplify paper measurements
chi = np.exp(2j*np.pi/3)
alpha_0 = 0
alpha_1 = np.pi/3
beta_0 = -np.pi/6
beta_1 = np.pi/6

# States
gamma = (np.sqrt(11) - np.sqrt(3))/2 # Maximum non-locality parameter
maxEntangledState = (1/np.sqrt(3))*(zero_zero + one_one + two_two)
anomalousState = (1/np.sqrt(2+(gamma**2)))*(zero_zero + gamma*one_one + two_two)
# Global variables
max_mixed_state = (1/9)*np.ones(36)
deterministic_matrix = matrix_of_det_strats() 

# Set up paper measurements
A_measurements, B_measurements = get_paper_measurements()
#print("Paper setup:")
#score_ME = SDP_opt(maxEntangledState, A_measurements, B_measurements)
#print("Max entangled =", score_ME)
#score_anom = SDP_opt(anomalousState, A_measurements, B_measurements)
#print("Anomalous =", score_anom)

# ---------------------------------- Main code -------------------------------

no_measures = int(1e2)
no_batches = 1 
epsilon_max_entangled = np.zeros(no_measures)
epsilon_anomalous = np.zeros(no_measures)
epsilon_max_entangled_all = np.array([])
epsilon_anomalous_all =  np.array([])

no_games = int(no_measures/no_batches)

# Choose to generate new data or use old data
mode = 'Calculate' 
#mode = 'Load'
if mode == 'Calculate':
    
    if __name__ == '__main__':
        no_cores = mp.cpu_count() - 1
        no_games_per_core = int(no_games/no_cores)
        
        random_seeds = random_numbers(no_cores*no_batches)
        inputs = [(no_games_per_core, seed) for seed in random_seeds]
        print("Number of unique random seeds = " + str(len(list(zip(*inputs))[1])))
        print("Number of cores = " + str(no_cores))
        print("Number of batches = " + str(no_batches) + "\n")
        
        for i in range(no_batches):
            parallel_data_production(no_games, i)
        
        for batch_no in range(no_batches):
            with open("ME_out/max entangled epsilons, no games {}, batch {}.npy".format(no_games, batch_no), "rb") as f:
                epsilon_max_entangled = np.load(f)
                epsilon_max_entangled_all = np.append(epsilon_max_entangled_all, epsilon_max_entangled)
                f.close()
            with open("anom_out/anomalous epsilons, no games {}, batch {}.npy".format(no_games, batch_no), "rb") as f:
                epsilon_anomalous = np.load(f)
                epsilon_anomalous_all = np.append(epsilon_anomalous_all, epsilon_anomalous)
                f.close()
                
        with open("max entangled epsilons, no games {}, batch all.npy".format(no_measures), "wb") as f:
            np.save(f, epsilon_max_entangled_all)
            f.close()
        with open("anomalous epsilons, no games {}, batch all.npy".format(no_measures), "wb") as f:
            np.save(f, epsilon_anomalous_all)
            f.close()
        
        no_games = no_measures
        write_stats("Robustness, no games {}, batch all.txt".format(no_measures), epsilon_max_entangled_all, epsilon_anomalous_all)
        plot(epsilon_max_entangled_all, epsilon_anomalous_all, no_measures, savename="Robustness")
    
    
    




