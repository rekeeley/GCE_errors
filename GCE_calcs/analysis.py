import numpy as np
import scipy.optimize as scop
from scipy import integrate

def poisson_log_like(k,mu):
    return k*np.log(mu) - mu - k*np.log(k) + k

def gauss_log_like(k,mu):
    return (k-mu)**2/(2*k)