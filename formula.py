import math
from scipy.special import binom

def regions_formula(c_bias, c_grad, K, N, n_in, axis_min, axis_max):
    T = 2**5 * c_bias * c_grad
    K_coef = binom(K * n_in, 2 * n_in)
    vol = (axis_max - axis_min)**n_in
    return ((T * K * N)**n_in * K_coef * vol) / math.factorial(n_in)

def regions_predicted_growth_with_K(K, N, n_in):
    K_coef = binom(K * n_in, 2 * n_in)
    return int(((K * N)**n_in * K_coef) / math.factorial(n_in))

def regions_predicted_growth(N, n_in):
    return int(((N)**n_in) / math.factorial(n_in))

def db_formula(c_bias, c_grad, K, N, n_in, axis_min, axis_max):
        T = 2**4 * c_bias * c_grad
        K_coef = binom(K * (n_in - 1), 2 * (n_in - 1))
        vol = (axis_max - axis_min)**n_in
        return ((T)**n_in * (2 * K * N)**(n_in - 1) * K_coef * vol) / math.factorial(n_in - 1)

def db_predicted_growth_with_K(K, N, n_in):
    K_coef = binom(K * (n_in - 1), 2 * (n_in - 1))
    return int(((K * N)**(n_in - 1) * K_coef) / math.factorial(n_in - 1))

def db_predicted_growth(N, n_in):
    return int(((N)**(n_in - 1)) / math.factorial(n_in - 1))
