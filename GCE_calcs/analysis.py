import numpy as np
import scipy.optimize as scop
from scipy import integrate
from scipy import stats
from scipy import interpolate


def poisson_log_like(k,mu):
    return k*np.log(mu) - mu - k*np.log(k) + k

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

def dwarf_delta_log_like(espectra,like_name):
    data = np.loadtxt('release-01-00-02/'+like_name+'.txt', unpack=True)
    delta_log_like = np.zeros(espectra.shape)
    for i in range(espectra.shape[3]):
        istart = i*25
        iend = istart+25
        # divide by 1000 to convert from MeV to GeV
        f = interpolate.interp1d(data[2,istart:iend]/1000.,data[3,istart:iend],kind='linear',bounds_error=False,fill_value='extrapolate')
        delta_log_like[:,:,:,i] = f(espectra[:,:,:,i])    
    return delta_log_like
    
def dwarf_delta_log_like_log_parab(espectra,like_name):
    data = np.loadtxt('release-01-00-02/'+like_name+'.txt', unpack=True)
    delta_log_like = np.zeros(espectra.shape)
    for i in range(espectra.shape[4]):
        istart = i*25
        iend = istart+25
        # divide by 1000 to convert from MeV to GeV
        f = interpolate.interp1d(data[2,istart:iend]/1000.,data[3,istart:iend],kind='linear',bounds_error=False,fill_value='extrapolate')
        delta_log_like[:,:,:,:,i] = f(espectra[:,:,:,:,i])    
    return delta_log_like    
