import numpy as np
import scipy.optimize as scop
from scipy import integrate
from scipy import stats


def poisson_log_like(k,mu):
    return k*np.log(mu) - mu - k*np.log(k) + k

def poisson_like(k,mu):
    return mu**k * np.exp(-mu) / exp(k*np.log(k) - k)

def gauss_log_like(k,mu):
    print k.shape
    print mu.shape
    return -(k-mu)**2/(2*k) - 0.5*(len(k)*np.log(2*np.pi) + np.log(np.prod(k)))

def gauss_log_like_unnorm(k,mu):
    return -(k-mu)**2/(2*k)

def get_J_log_prior_fast(J):
    sigma_rho = 0.08
    mu_rho = 0.28
    norm = -0.5*np.log(2*np.pi*sigma_rho**2) + np.log(0.5*(J/2e23)**-0.5)
    chi_sqrd = -0.5*( np.sqrt(J/2.e23) - mu_rho)**2/(sigma_rho)**2
    return norm + chi_sqrd

def get_J_prior_MC(J_range):
    # J_range is a vector of J values where the KDE is to be evaluated
    rho = np.random.normal(0.28,0.08,80000)
    J = 2e23*rho**2
    J_kde = stats.gaussian_kde(J)
    return J_kde(J_range)
