import numpy as np
import scipy.optimize as scop
from scipy import integrate
from scipy import stats
from scipy import interpolate


def poisson_log_like(k,mu):
    return k*np.log(mu) - mu - k*np.log(k) + k

def density(r,scale_radius,  gamma): #NFW density profile
    #r is the physical radius in kpc
	solar_radius = 8.25
	R = r/solar_radius#R / solar radius
	return pow(R,-gamma)*pow((1 + R*solar_radius/scale_radius)/(1+solar_radius/scale_radius),gamma-3)


def get_J_prior_MC(J_range):
    # J_range is a vector of J values where the KDE is to be evaluated
    rho = np.random.normal(0.28,0.08,200000) #Zhang et al 1209.0256
    J = 2.e23*rho**2
    J_kde = stats.gaussian_kde(J)
    return J_kde(J_range)

def get_J_prior_MC_2(J_range):
    # J_range is a vector of J values where the KDE is to be evaluated
    rho = np.random.normal(0.42,0.03,200000) #Pato et al values 1504.06324v2
    J = 2.e23*rho**2
    J_kde = stats.gaussian_kde(J)
    return J_kde(J_range)

def get_J_prior_MC_3(J_range):
    # J_range is a vector of J values where the KDE is to be evaluated
    rho = np.random.normal(0.49,0.13,200000) #McKee et al values 1509.05334v1
    J = 2.e23*rho**2
    J_kde = stats.gaussian_kde(J)
    return J_kde(J_range)


def get_J_prior_dwarf(J,mu,sigma):
    exponent = np.random.normal(mu,sigma,200000)
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
