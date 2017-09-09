import numpy as np
import scipy.optimize as scop
from scipy import integrate
from scipy import stats
from scipy import interpolate


def poisson_log_like(k,mu):
    return k*np.log(mu) - mu - k*np.log(k) + k

def get_J_prior_dwarf(J,mu,sigma):
    exponent = np.random.normal(mu,sigma,100000)
    J_kde_exp = stats.gaussian_kde(exponent)
    return J_kde_exp(np.log10(J))

def get_msp_prior(norm_range,mu,sigma,dist_mu,dist_sigma):
    dist = np.random.normal(dist_mu,dist_sigma,200000)
    SM = 10.**np.random.normal(np.log10(mu),0.3,200000)
    SM_kde = stats.gaussian_kde(np.log10(SM/dist**2./4./np.pi))
    return SM_kde(np.log10(norm_range))

def get_SIDM_prior(norm_range, SM_mu, SM_sigma):
    rho = np.random.normal(0.28,0.08,1000000)
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

def GCE_delta_log_like(espectra,model_name,trunc):
    if model_name == 'MIT':
        loaded = np.load('data/background/LL_inten_GCE.npz')
        loglike = loaded['LL']
        nflux = loaded['inten']*0.06854 #multiply by the number of steradians in the 15x15 ROI
    else:
        loaded = np.load('data/background/GC_'+model_name+'.npz')
        exposure = loaded['exposure']
        loglike = -loaded['LL']# the loaded quantities are actual negative log probabilites so thats why theres a negative
        exposure = np.tile(exposure[:,np.newaxis],(1,loglike.shape[1]))
        nflux = loaded['counts']/exposure
        nflux = nflux[trunc:,:]
        loglike = loglike[trunc:,:]
    # 4*pi *15*15/41252.96
    delta_log_like = np.zeros(espectra.shape)
    for i in range(espectra.shape[3]):
        f = interpolate.interp1d(nflux[i,:],loglike[i,:]-loglike[i,:].max(),kind='linear',bounds_error=False,fill_value='extrapolate')
        delta_log_like[:,:,:,i] = f(espectra[:,:,:,i])    
    return delta_log_like

def GCE_delta_log_like_msp(espectra,model_name,trunc):
    loaded = np.load('data/background/GC_'+model_name+'.npz')
    exposure = loaded['exposure']
    loglike = -loaded['LL']# the loaded quantities are actual negative log probabilites so thats why theres a negative
    exposure = np.tile(exposure[:,np.newaxis],(1,loglike.shape[1]))
    nflux = loaded['counts']/exposure
    nflux = nflux[trunc:,:]
    loglike = loglike[trunc:,:]
    # 4*pi *15*15/41252.96
    delta_log_like = np.zeros(espectra.shape)
    for i in range(espectra.shape[4]):
        f = interpolate.interp1d(nflux[i,:],loglike[i,:]-loglike[i,:].max(),kind='linear',bounds_error=False,fill_value='extrapolate')
        delta_log_like[:,:,:,:,i] = f(espectra[:,:,:,:,i])
    return delta_log_like

def dwarf_delta_log_like(espectra,like_name):
    data = np.loadtxt('release-01-00-00/'+like_name+'.txt', unpack=True)
    delta_log_like = np.zeros(espectra.shape)
    for i in range(espectra.shape[3]):
        istart = i*25
        iend = istart+25
        # divide by 1000 to convert from MeV to GeV
        f = interpolate.interp1d(data[2,istart:iend]/1000.,data[3,istart:iend],kind='linear',bounds_error=False,fill_value='extrapolate')
        delta_log_like[:,:,:,i] = f(espectra[:,:,:,i])    
    return delta_log_like
    
def dwarf_delta_log_like_msp(espectra,like_name):
    data = np.loadtxt('release-01-00-02/'+like_name+'.txt', unpack=True)
    delta_log_like = np.zeros(espectra.shape)
    for i in range(espectra.shape[4]):
        istart = i*25
        iend = istart+25
        # divide by 1000 to convert from MeV to GeV
        f = interpolate.interp1d(data[2,istart:iend]/1000.,data[3,istart:iend],kind='linear',bounds_error=False,fill_value='extrapolate')
        delta_log_like[:,:,:,:,i] = f(espectra[:,:,:,:,i])    
    return delta_log_like    
