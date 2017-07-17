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

def get_J_prior_dwarf(J,mu,sigma):
    exponent = np.random.normal(mu,sigma,1000000)
    J_kde = stats.gaussian_kde(10**exponent)
    return J_kde(J)

def get_msp_prior(norm_range,mu,sigma,dist_mu,dist_sigma):
    dist = np.random.normal(dist_mu,dist_sigma,200000)
    SM = 10.**np.random.normal(np.log10(mu),0.3,200000)
    SM_kde = stats.gaussian_kde(SM/dist**2./4./np.pi)
    return SM_kde(norm_range)

def get_SIDM_prior(norm_range, SM_mu, SM_sigma):
    rho = np.random.normal(0.28,0.08,1000000) #Zhang et al 1209.0256
    J = 2.e23*rho**2
    SM = np.random.normal(1.0,SM_sigma/SM_mu,1000000)
    norm_kde = stats.gaussian_kde(J*SM)
    return norm_kde(norm_range)

def get_SIDM_dwarf_prior(norm_range, J_mu, J_sigma, SM_mu, SM_sigma):
    exponent = np.random.normal(J_mu, J_sigma, 200000)
    J = 10**exponent
    SM = np.random.normal(SM_mu, SM_sigma, 200000)
    norm_kde = stats.gaussian_kde(J*SM)
    return norm_kde(norm_range)

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
