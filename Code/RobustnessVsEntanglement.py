# Imports and required parameters

import numpy as np
import cvxpy as cp
import qutip as qp
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
import seaborn as sns
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
    state_samples = int(sys.argv[4])

except:
    no_dimensions = 2
    no_measurements = 2
    no_games = int(1e4)
    state_samples = 30

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
    # print(M_proj, qp.qeye(no_dimensions))
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
    # print(Alice_measurement, state)
    sigma = sigma_ax(Alice_measurement, state)
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
            # else:
            #    a_value = no_dimensions - 1
        # Have ai_value (subsection) & bj_value (pos. in subsec.)
        for j in range(no_measurements):
            if i == j:
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
    vec = np.zeros(n2 * d2)
    count = 0
    for i in range(no_measurements):
        for k in range(no_measurements):
            vec[count * d2 + mat[i, k]] = 1
            count += 1
    return vec


def matrix_of_det_strats():  # any dimension and number of measurements

    no_rows = n2 * d2
    no_cols = (no_dimensions ** no_measurements) ** 2
    matrix = np.zeros((no_rows, no_cols))
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

    for j in range((no_dimensions ** no_measurements) ** 2):
        vec = mat[:, j]
        if np.sum(vec) != n2:
            print("Error 1")
            return None

        for k in range(n2):  # Loop over segments
            temp = 0
            for l in range(d2):  # Loop over elements in the segments
                temp += vec[k * d2 + l]
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
    vector = np.zeros(n2 * d2)
    count = 0
    vec_count = 0

    for a in range(no_measurements):
        for b in range(no_measurements):
            for i in range(a * no_dimensions, (a + 1) * no_dimensions):
                for j in range(b * no_dimensions, (b + 1) * no_dimensions):
                    vector[count] = probability(state, Alice_measurements[i], Bob_measurements[j])
                    count += 1
            vector[(vec_count + 1) * d2 - 1] = 1 - np.sum(vector[(vec_count) * d2:(vec_count + 1) * d2 - 1])
            vec_count += 1

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
    # M = Model()

    I = cp.Variable(d2 * n2)  # q_vec[i] >= 0, sum(q_vec[i]) = 1

    # Convert state from computational basis to 36 probability vector
    state_vec = state_vector(state, Alice_measurements, Bob_measurements)

    # Construct the problem.
    objective = cp.Maximize((I @ state_vec - 1) / 2)

    # Set the remaining constraints
    constraints = [np.transpose(D) @ I >= -1, np.transpose(D) @ I <= 1]

    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.MOSEK)

    if result < 10 ** (-8):
        return 0

    return result


def SDP_opt_old(state, Alice_measurements, Bob_measurements):
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
    # M = Model()
    r = cp.Variable(nonneg=True)
    s = cp.Variable(((no_dimensions ** no_measurements) ** 2,), nonneg=True)  # q_vec[i] >= 0, sum(q_vec[i]) = 1
    q = cp.Variable(((no_dimensions ** no_measurements) ** 2,), nonneg=True)

    # Convert state from computational basis to 36 probability vector
    state_vec = state_vector(state, Alice_measurements, Bob_measurements)

    # Construct the problem.
    objective = cp.Minimize(r)
    # Set the remaining constraints
    constraints = [state_vec + (D @ s) == (D @ q),
                   cp.sum(q) == 1 + r, cp.sum(s) == r]

    prob = cp.Problem(objective, constraints)

    return prob.solve(solver=cp.MOSEK)


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
        f.write(
            "Number of dimensions = {}, number of measurements per party = {}\n".format(no_dimensions, no_measurements))
        f.write("Number of games = {}\n\n".format(no_games))
        f.write("Max entangled state:\n\n")
        f.write("Volume of nonlocality = {}\n\n".format(volume_ME))
        f.write("Total dataset:\n")
        f.write("Lowest score = {}\n".format(min(robustness_ME)))
        f.write("Highest score = {}\n".format(max(robustness_ME)))
        f.write("Mean score = {} ± {}\n".format(np.mean(robustness_ME),
                                                3 * np.mean(robustness_ME) / np.sqrt(len(robustness_ME) - 1)))
        f.write("Median score = {}\n".format(np.median(robustness_ME)))
        f.write("STD of scores = {}\n".format(np.std(robustness_ME)))
        # f.write("Variance of scores = {}\n\n".format(np.var(robustness_ME)))

        f.write("Conditional dataset (remove zero r (epsilons)):\n")
        f.write("Lowest score = {}\n".format(min(nonzero_robustness_ME)))
        f.write("Mean score = {} ± {}\n".format(np.mean(nonzero_robustness_ME),
                                                3 * np.mean(nonzero_robustness_ME) / np.sqrt(
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
    # if ylabel == "NLV":
    #    y_err = 3* y_err * np.sqrt((state_samples-1)/(no_games-1))

    ax.plot(entanglement, y, 'o', color='tab:purple')
    ax.errorbar(entanglement, y, yerr=y_err, fmt='o')
    ax.set_xlabel("Entanglement", size=16)
    ax.set_ylabel(ylabel, size=16)

    plt.savefig(title + ".pdf")
    plt.show()


def plot_two_axis(xdata, y1, y2, y1_err, y2_err,
                  y_label, y2_label, title=None, savename=None, log=False):
    # Remove the if-statement for new data
    # if ylabel == "NLV":
    #     y_err = 3* y_err * np.sqrt((state_samples-1)/(no_games-1))

    # Order
    entanglement_index = np.argsort(entanglement)
    xdata = [xdata[i] for i in entanglement_index]
    y1 = [y1[i] for i in entanglement_index]
    y2 = [y2[i] for i in entanglement_index]
    y1_err = [y1_err[i] for i in entanglement_index]
    y2_err = [y2_err[i] for i in entanglement_index]

    # Get rid of zero elements to avoid taking the log of 0
    y1 = np.delete(y1, np.where(y1 == 0))
    y2 = np.delete(y2, np.where(y2 == 0))
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
        ax1.semilogy(xdata, y1, 'o', color='tab:red', label="Mean", linestyle="None")
    else:
        ax1.plot(xdata, y1, 'o', color='tab:red', label="Mean", linestyle="None")

    ax1.errorbar(xdata, y1, y1_err, color='tab:red', linestyle="None")

    # Do second y axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_label)
    if log == True:
        ax2.semilogy(xdata, y2, 'o', color='tab:blue', label="Cond mean", linestyle="None")
    else:
        ax2.plot(xdata, y2, 'o', color='tab:blue', label="Cond mean", linestyle="None")

    ax2.errorbar(xdata, y2, y2_err, color='tab:blue', linestyle="None")

    # Combine legend data
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc=2)

    if title != None:
        plt.title(title)

    if savename != None:
        plt.savefig(savename + '.pdf')

    plt.show()


def generateQubit(theta):
    state = np.cos(theta) * zero_zero + np.sin(theta) * one_one

    return state


def Von_Neumann_Entanglement(theta):
    lambda_0 = np.cos(theta)
    lambda_1 = np.sin(theta)

    if lambda_0 == 0 or lambda_1 == 0:
        return 0

    else:
        return -(lambda_0) * np.log(lambda_0) - (lambda_1) * np.log(lambda_1)


def Robustness_Entanglement(theta):
    return 2*np.sin(theta)*np.cos(theta)


def calculate(state_samples, seed):
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

    mean_robustness = np.zeros(state_samples)
    cond_mean_robustness = np.zeros(state_samples)
    NLV = np.zeros(state_samples)

    mean_robustness_err = np.zeros(state_samples)
    cond_mean_robustness_err = np.zeros(state_samples)
    NLV_err = np.zeros(state_samples)

    entanglement = np.zeros(state_samples)

    for k in tqdm(range(state_samples), desc="Generating robustness..."):
        theta = (np.pi / 4) * np.random.random()
        state = generateQubit(theta)
        entanglement[k] = Robustness_Entanglement(theta)

        robustness = np.zeros(no_games)
        for i in range(no_games):
            # Get random projective measurements
            A_measurements = get_random_measurements()
            B_measurements = get_random_measurements()

            # Get max entangled robustness
            robustness[i] = SDP_opt(state, A_measurements, B_measurements)

        nonzero_robustness = np.delete(robustness, np.where(robustness < 1e-8))

        mean_robustness[k] = np.mean(robustness)
        NLV[k] = len(nonzero_robustness) / len(robustness)

        mean_robustness_err[k] = 3 * np.mean(robustness) / np.sqrt(len(robustness) - 1)

        if len(nonzero_robustness) == 0 or len(nonzero_robustness) == 1:
            cond_mean_robustness[k] = 0
            cond_mean_robustness_err[k] = 0
            NLV_err[k] = 0

        else:
            cond_mean_robustness[k] = np.mean(nonzero_robustness)
            cond_mean_robustness_err[k] = 3 * np.mean(nonzero_robustness) / np.sqrt(len(nonzero_robustness) - 1)
            NLV_err[k] = 3 * np.sqrt((NLV[k] - NLV[k] ** 2) / (len(nonzero_robustness) - 1))

    return entanglement, mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err


def random_numbers(no):
    rand_nos = np.random.randint(0, int(2 ** 32 - 1), dtype=np.int64, size=no)
    if len(np.unique(rand_nos)) != no:
        random_numbers(no)

    return rand_nos


def save(entanglement, mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err,
         NLV_err, batch_no):
    with open("../Data/Nonlocality vs Entanglement/Entanglement, no_games {}, state samples {}, d{}, m{} batch {}.npy"
                      .format(no_games, state_samples, no_dimensions, no_measurements, batch_no), "wb") as f:
        np.save(f, entanglement)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Mean robustness, no_games {}, state samples {}, d{}, m{} batch {}.npy"
                      .format(no_games, state_samples, no_dimensions, no_measurements, batch_no), "wb") as f:
        np.save(f, mean_robustness)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Conditional mean robustness, no_games {}, state samples {}, d{}, m{} batch {}.npy"
                      .format(no_games, state_samples, no_dimensions, no_measurements, batch_no), "wb") as f:
        np.save(f, cond_mean_robustness)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/NLV, no_games {}, state samples {}, d{}, m{} batch {}.npy"
                      .format(no_games, state_samples, no_dimensions, no_measurements, batch_no), "wb") as f:
        np.save(f, NLV)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Mean robustness error, no_games {}, state samples {}, d{}, m{} batch {}.npy"
                      .format(no_games, state_samples, no_dimensions, no_measurements, batch_no), "wb") as f:
        np.save(f, mean_robustness_err)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Conditional mean robustness error, no_games {}, state samples {}, d{}, m{} batch {}.npy"
                      .format(no_games, state_samples, no_dimensions, no_measurements ,batch_no), "wb") as f:
        np.save(f, cond_mean_robustness_err)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/NLV error, no_games {}, state samples {}, d{}, m{} batch {}.npy".format(no_games, state_samples, no_dimensions, no_measurements, batch_no),
              "wb") as f:
        np.save(f, NLV_err)
        f.close()


def save_all(entanglement, mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err,
             NLV_err):
    with open("../Data/Nonlocality vs Entanglement/Entanglement, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements), "wb") as f:
        np.save(f, entanglement)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Mean robustness, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements), "wb") as f:
        np.save(f, mean_robustness)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Conditional mean robustness, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements),
              "wb") as f:
        np.save(f, cond_mean_robustness)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/NLV, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements), "wb") as f:
        np.save(f, NLV)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Mean robustness error, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements),
              "wb") as f:
        np.save(f, mean_robustness_err)
        f.close()

    with open("../Data//Nonlocality vs Entanglement/Conditional mean robustness error, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements),
            "wb") as f:
        np.save(f, cond_mean_robustness_err)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/NLV error, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements), "wb") as f:
        np.save(f, NLV_err)
        f.close()


def load(batch_no):
    with open("../Data/Nonlocality vs Entanglement/Entanglement, no_games {}, state samples {}, d{}, m{} batch {}.npy".format(no_games, state_samples, no_dimensions, no_measurements, batch_no),
              "rb") as f:
        entanglement = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Mean robustness, no_games {}, state samples {}, d{}, m{} batch {}.npy".format(no_games, state_samples, no_dimensions, no_measurements, batch_no),
              "rb") as f:
        mean_robustness = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Conditional mean robustness, no_games {}, state samples {}, d{}, m{} batch {}.npy".format(no_games, state_samples, no_dimensions, no_measurements,
                      batch_no), "rb") as f:
        cond_mean_robustness = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/NLV, no_games {}, state samples {}, d{}, m{} batch {}.npy".format(no_games, state_samples, no_dimensions, no_measurements, batch_no), "rb") as f:
        NLV = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Mean robustness error, no_games {}, state samples {}, d{}, m{} batch {}.npy".format(no_games, state_samples, no_dimensions, no_measurements,
                batch_no), "rb") as f:
        mean_robustness_err = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Conditional mean robustness error, no_games {}, state samples {}, d{}, m{} batch {}.npy".format(no_games, state_samples, no_dimensions, no_measurements, batch_no), "rb") as f:
        cond_mean_robustness_err = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/NLV error, no_games {}, state samples {}, d{}, m{} batch {}.npy".format(no_games, state_samples, no_dimensions, no_measurements, batch_no),
              "rb") as f:
        NLV_err = np.load(f)
        f.close()

    return entanglement, mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err


def load_all():
    with open("../Data/Nonlocality vs Entanglement/Entanglement, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements), "rb") as f:
        entanglement = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Mean robustness, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements), "rb") as f:
        mean_robustness = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Conditional mean robustness, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements),
              "rb") as f:
        cond_mean_robustness = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/NLV, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements), "rb") as f:
        NLV = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Mean robustness error, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements),
              "rb") as f:
        mean_robustness_err = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/Conditional mean robustness error, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements),
            "rb") as f:
        cond_mean_robustness_err = np.load(f)
        f.close()

    with open("../Data/Nonlocality vs Entanglement/NLV error, no_games {}, state samples {}, d{}, m{} all.npy".format(no_games, state_samples, no_dimensions, no_measurements), "rb") as f:
        NLV_err = np.load(f)
        f.close()

    return entanglement, mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err


def parallel_data_production(batch_no=0):
    with mp.Pool() as pool:
        results = pool.starmap(calculate,
                               inputs[no_cores * batch_no:no_cores * (batch_no + 1)])  # [no_games_per_core]*no_cores)

    entanglement = np.concatenate([results[n][0] for n in range(no_cores)])
    mean_robustness = np.concatenate([results[n][1] for n in range(no_cores)])
    cond_mean_robustness = np.concatenate([results[n][2] for n in range(no_cores)])
    NLV = np.concatenate([results[n][3] for n in range(no_cores)])
    mean_robustness_err = np.concatenate([results[n][4] for n in range(no_cores)])
    cond_mean_robustness_err = np.concatenate([results[n][5] for n in range(no_cores)])
    NLV_err = np.concatenate([results[n][6] for n in range(no_cores)])

    save(entanglement, mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err,
         NLV_err, batch_no)

    # write_stats("Robustness, no games {}, batch number {}.txt".format(no_games, batch_no), batch_no)


def order(entanglement, ydata):
    entanglement_index = np.argsort(entanglement)
    entanglement = [entanglement[i] for i in entanglement_index]
    ydata = [ydata[i] for i in entanglement_index]

    return entanglement, ydata

zero = qp.basis(2, 0)
one = qp.basis(2, 1)
zero_zero = qp.tensor(zero, zero)
one_one = qp.tensor(one, one)


no_batches = 1
state_samples_per_batch = int(state_samples / no_batches)

# Conversions
d2 = no_dimensions ** 2
n2 = no_measurements ** 2

D = matrix_of_det_strats()

entanglement_all = np.array([])
mean_robustness_all = np.array([])
cond_mean_robustness_all = np.array([])
NLV_all = np.array([])
mean_robustness_err_all = np.array([])
cond_mean_robustness_err_all = np.array([])
NLV_err_all = np.array([])

mode = 'load'

if mode == 'calculate':

    if __name__ == '__main__':
        no_cores = mp.cpu_count() - 1
        state_samples_per_core = int(state_samples_per_batch / no_cores)

        random_seeds = random_numbers(no_cores * no_batches)
        inputs = [(state_samples_per_core, seed) for seed in random_seeds]
        print("Number of unique random seeds = " + str(len(list(zip(*inputs))[1])))
        print("Number of cores = " + str(no_cores))
        print("Number of batches = " + str(no_batches))
        print("d={}, n={}, no games={}, no state samples={}\n".format(
            no_dimensions, no_measurements, no_games, state_samples))

        for i in range(no_batches):
            parallel_data_production(i)

        for batch_no in range(no_batches):
            entanglement, mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err = load(
                batch_no)

            entanglement_all = np.append(entanglement_all, entanglement)
            mean_robustness_all = np.append(mean_robustness_all, mean_robustness)
            cond_mean_robustness_all = np.append(cond_mean_robustness_all, cond_mean_robustness)
            NLV_all = np.append(NLV_all, NLV)
            mean_robustness_err_all = np.append(mean_robustness_err_all, mean_robustness_err)
            cond_mean_robustness_err_all = np.append(cond_mean_robustness_err_all, cond_mean_robustness_err)
            NLV_err_all = np.append(NLV_err_all, NLV_err)

        save_all(entanglement_all, mean_robustness_all, cond_mean_robustness_all, NLV_all, mean_robustness_err_all,
                 cond_mean_robustness_err_all, NLV_err_all)

if mode == 'load':
    entanglement, mean_robustness, cond_mean_robustness, NLV, mean_robustness_err, cond_mean_robustness_err, NLV_err = load_all()


    def new_plot(xdata, y1, y2, y1_err, y2_err,
                       y_label, x_label='Robustness Entanglement', title=None, savename=None, log=False):

        entanglement_index = np.argsort(xdata)
        xdata = [xdata[i] for i in entanglement_index]
        y1 = [y1[i] for i in entanglement_index]
        y2 = [y2[i] for i in entanglement_index]
        y1_err = [y1_err[i] for i in entanglement_index]
        y2_err = [y2_err[i] for i in entanglement_index]

        # Get rid of zero elements to avoid taking the log of 0
        y1 = np.delete(y1, np.where(y1 == 0))
        y2 = np.delete(y2, np.where(y2 == 0))
        length = min(len(y1), len(y2))

        y1 = y1[-length::]
        y2 = y2[-length::]

        xdata = xdata[-length::]
        y1_err = y1_err[-length::]
        y2_err = y2_err[-length::]

        fig, ax = plt.subplots()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.plot(xdata, y1, 'o', color='tab:red', label="Mean")  # , linestyle = "None")

        ax.errorbar(xdata, y1, y1_err, color='tab:red')  # , linestyle = "None")

        # Second dataset
        ax.plot(xdata, y2, 'o', color='tab:blue', label="Cond mean")  # , linestyle = "None")
        ax.errorbar(xdata, y2, y2_err, color='tab:blue')  # , linestyle = "None")

        if log==True:
            #ax.set_yscale('log')
            #ax.set_xscale('log')
            ax.loglog()
            a=2

        # Legend, title, and saving
        ax.legend(loc=2)
        ax.grid()
        #if title != None:
        #    plt.title(title)
        if savename != None:
            plt.savefig("../Plots/RobustnessVsEntanglement/"+savename + '.pdf')

        plt.show()


    new_plot(entanglement, mean_robustness, cond_mean_robustness, mean_robustness_err, cond_mean_robustness_err,
             "Robustness", "Robustness Entanglement","Normal Plot", savename=f"RvsE, d={no_dimensions}, m={no_measurements}")  # "Robustness vs number of measurements, d={}".format(no_dimensions))

    log_entanglement = np.log(entanglement+1)
    log_mean_robustness = np.log(mean_robustness+1)
    log_cond_mean_robustness = np.log(cond_mean_robustness+1)
    new_plot(log_entanglement, log_mean_robustness, log_cond_mean_robustness, mean_robustness_err, cond_mean_robustness_err,
                   "log(1+r)", r"log(1+$R_E$)", "Log Plot", savename=f"RvsE, d={no_dimensions}, m={no_measurements}, log plot", log=True)#"Robustness vs number of measurements, d={}".format(no_dimensions))
    































