import numpy as np
import scipy.optimize as scop
from scipy import integrate
from scipy import stats


def poisson_log_like(k,mu):
    return k*np.log(mu) - mu - k*np.log(k) + k

def poisson_log_like_unnorm(k,mu):
    return k*np.log(mu) - mu

def poisson_like(k,mu):
    return mu**k * np.exp(-mu) / np.exp(k*np.log(k) - k)

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
    J = 2.e23*rho**2
    J_kde = stats.gaussian_kde(J)
    return J_kde(J_range)

def get_J_prior_dwarf(J,mu,sigma):
    exponent = np.random.normal(mu,sigma,80000)
    J_kde = stats.gaussian_kde(10**exponent)
    return J_kde(J)

def get_J_prior_Bootes(J):
    log_J_mean = 18.8
    log_J_sigma = 0.22
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_CanesVenaticiII(J):
    log_J_mean = 17.9
    log_J_sigma = 0.25
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_Carina(J):
    log_J_mean = 18.1
    log_J_sigma = 0.23
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_ComaBerenices(J):
    log_J_mean = 19.0
    log_J_sigma = 0.25
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_Draco(J):
    log_J_mean = 18.8
    log_J_sigma = 0.16
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_Fornax(J):
    log_J_mean = 18.2
    log_J_sigma = 0.21
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_Hercules(J):
    log_J_mean = 18.1
    log_J_sigma = 0.25
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_LeoII(J):
    log_J_mean = 17.6
    log_J_sigma = 0.18
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_LeoIV(J):
    log_J_mean = 17.9
    log_J_sigma = 0.28
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_Sculptor(J):
    log_J_mean = 18.6
    log_J_sigma = 0.18
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_SegueI(J):
    log_J_mean = 19.5
    log_J_sigma = 0.29
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_Sextans(J):
    log_J_mean = 18.4
    log_J_sigma = 0.27
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_UrsaMajorII(J):
    log_J_mean = 19.3
    log_J_sigma = 0.28
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_UrsaMinor(J):
    log_J_mean = 18.8
    log_J_sigma = 0.19
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_WillmanI(J):
    log_J_mean = 19.1
    log_J_sigma = 0.31
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_CanesVenaticiI(J):
    log_J_mean = 17.7
    log_J_sigma = 0.26
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_LeoI(J):
    log_J_mean = 17.7
    log_J_sigma = 0.18
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)

def get_J_prior_UrsaMajorI(J):
    log_J_mean = 18.3
    log_J_sigma = 0.24
    exponent = np.random.normal(log_J_mean,log_J_sigma,80000)
    J_kde = stats.gaussian_kde(10.**exponent)
    return J_kde(J)
