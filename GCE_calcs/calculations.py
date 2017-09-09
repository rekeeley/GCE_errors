import numpy as np
import scipy.optimize as scop
from scipy import integrate
from scipy.special import erf, gammainc


def get_mu(background,exposure,num_spec,J,sigma,mass):
    #should return an array of shape (n_cross,n_J,n_mass,n_spec)
    #background, exposure should be vectors of len n_spec: the number of energy bins
    #num_spec is an array with shape n_mass,n_spec, it is the binned spectra for the various energy bins, for each mass
    #J, sigma, mass are all vectors with their corresponding length
    n_sigma = len(sigma)
    n_mass = len(mass)
    n_J = len(J)
    n_spec = len(background)
    background = np.tile(background[np.newaxis,np.newaxis,np.newaxis,:],(n_sigma,n_J,n_mass,1))
    exposure = np.tile(exposure[np.newaxis,np.newaxis,np.newaxis,:],(n_sigma,n_J,n_mass,1))
    J = np.tile(J[np.newaxis,:,np.newaxis,np.newaxis],(n_sigma,1,n_mass,n_spec))
    mass = np.tile(mass[np.newaxis,np.newaxis,:,np.newaxis],(n_sigma,n_J,1,n_spec))
    sigma = np.tile(sigma[:,np.newaxis,np.newaxis,np.newaxis],(1,n_J,n_mass,n_spec))
    num_spec = np.tile(num_spec[np.newaxis,np.newaxis,:],(n_sigma,n_J,1,1))
    return background + exposure*J*sigma*num_spec/(8.*np.pi*mass**2)

def get_eflux(e_spec,J,sigma,mass):
    #should return an array of shape (n_cross,n_J,n_mass,n_spec)
    #background, exposure should be vectors of len n_spec: the number of energy bins
    #e_spec is an array with shape n_mass,n_spec, it is the binned spectra for the various energy bins, for each mass
    #J, sigma, mass are all vectors with their corresponding length
    n_sigma = len(sigma)
    n_mass = len(mass)
    n_J = len(J)
    n_mass2,n_spec = e_spec.shape
    assert n_mass ==n_mass2, 'the mass and spectra arrays are incompatible'
    J = np.tile(J[np.newaxis,:,np.newaxis,np.newaxis],(n_sigma,1,n_mass,n_spec))
    mass = np.tile(mass[np.newaxis,np.newaxis,:,np.newaxis],(n_sigma,n_J,1,n_spec))
    sigma = np.tile(sigma[:,np.newaxis,np.newaxis,np.newaxis],(1,n_J,n_mass,n_spec))
    e_spec = np.tile(e_spec[np.newaxis,np.newaxis,:],(n_sigma,n_J,1,1))
    eflux = J*sigma*e_spec/(8.*np.pi*mass**2)
    return eflux


def get_mu_SIDM(background,exposure,num_spec,J,sigma,mass):
    #should return an array of shape (n_cross,n_J,n_mass,n_spec)
    #background, exposure should be vectors of len n_spec: the number of energy bins
    #num_spec is an array with shape n_mass,n_spec, it is the binned spectra for the various energy bins, for each mass
    #J, sigma, mass are all vectors with their corresponding length
    n_gamma = 5.e3 # the number density of photons (NOT the length of some gamma vector)
    l_GC = 3.086e21
    sigma_t = 6.6524e-25  #the Thompson cross-section in cm^2
    n_sigma = len(sigma)
    n_mass = len(mass)
    n_J = len(J)
    n_spec = len(background)
    background = np.tile(background[np.newaxis,np.newaxis,np.newaxis,:],(n_sigma,n_J,n_mass,1))
    exposure = np.tile(exposure[np.newaxis,np.newaxis,np.newaxis,:],(n_sigma,n_J,n_mass,1))
    J = np.tile(J[np.newaxis,:,np.newaxis,np.newaxis],(n_sigma,1,n_mass,n_spec))
    mass = np.tile(mass[np.newaxis,np.newaxis,:,np.newaxis],(n_sigma,n_J,1,n_spec))
    sigma = np.tile(sigma[:,np.newaxis,np.newaxis,np.newaxis],(1,n_J,n_mass,n_spec))
    num_spec = np.tile(num_spec[np.newaxis,np.newaxis,:],(n_sigma,n_J,1,1))
    return background + exposure*J*n_gamma*sigma_t*sigma*l_GC*num_spec/(16.*np.pi*mass**2)

def get_eflux_SIDM(e_spec,J,sigma,mass):
    #should return an array of shape (n_cross,n_J,n_mass,n_spec)
    #background, exposure should be vectors of len n_spec: the number of energy bins
    #e_spec is an array with shape n_mass,n_spec, it is the binned spectra for the various energy bins, for each mass
    #J, sigma, mass are all vectors with their corresponding length
    n_gamma = 5.e1 # the number density of photons (NOT the length of some gamma vector) in 1/cm^3
    l_dwarf = 3.086e21 # the length scale of the dwarf 1 kpc-ish
    sigma_t = 6.6524e-25  #the Thompson cross-section in cm^2
    n_sigma = len(sigma)
    n_mass = len(mass)
    n_J = len(J)
    n_mass2,n_spec = e_spec.shape
    assert n_mass ==n_mass2, 'the mass and spectra arrays are incompatible'
    J = np.tile(J[np.newaxis,:,np.newaxis,np.newaxis],(n_sigma,1,n_mass,n_spec))
    mass = np.tile(mass[np.newaxis,np.newaxis,:,np.newaxis],(n_sigma,n_J,1,n_spec))
    sigma = np.tile(sigma[:,np.newaxis,np.newaxis,np.newaxis],(1,n_J,n_mass,n_spec))
    e_spec = np.tile(e_spec[np.newaxis,np.newaxis,:],(n_sigma,n_J,1,1))
    eflux = J*n_gamma*sigma_t*sigma*l_dwarf*e_spec/(16.*np.pi*mass**2)
    return eflux    



def get_mu_log_parab(background,exposure,num_spec):
    #each of the inputs should be arrays of the shape (n_gfpsm, n_SM, n_alpha, n_beta, n_spec)
    return background + exposure*num_spec

def get_spec_log_parab(gfpsm,SM,alpha,beta,Eb,emax,emin):
    #emax is a vector of len n_spec, the maxima energy of each bin
    #emin is a vector of len n_spec, the minima energy of each bin
    #E_b is a scalar, the scale energy of the spectra  shpuld be greater than the max for alpha,beta >0
    #beta is a vector of len n_beta, the parameter for exponential cutoff,
    #alpha is a vector of len n_alpha, the parameter for the power law part of the spectra
    #N0 is a vector of len n_N0, the parameter for the normalization
    n_spec = len(emax)
    n_beta = len(beta)
    n_alpha = len(alpha)
    n_gfpsm = len(gfpsm)
    n_SM = len(SM)
    gfpsm = np.tile(gfpsm[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],(1,n_SM,n_alpha,n_beta,n_spec))
    SM   =  np.tile(   SM[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis],(n_gfpsm,1,n_alpha,n_beta,n_spec))
    alpha = np.tile(alpha[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis],(n_gfpsm,n_SM,1,n_beta,n_spec))
    beta  = np.tile( beta[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis],(n_gfpsm,n_SM,n_alpha,1,n_spec))
    emax  = np.tile( emax[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_gfpsm,n_SM,n_alpha,n_beta,1))
    emin  = np.tile( emin[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_gfpsm,n_SM,n_alpha,n_beta,1))
    norm = erf(0.5*(alpha-1)/np.sqrt(beta)+np.sqrt(beta)*np.log(50./Eb)) - erf(0.5*(alpha-1)/np.sqrt(beta)+np.sqrt(beta)*np.log(0.200/Eb))
    #this norm term is to try and break any degeneracy between the normalization and the spectral parameters.  This seems more important in the zoa case since we are trying to attach physical meaning to this normalization
    return gfpsm*SM*(erf(0.5*(alpha-1)/np.sqrt(beta)+np.sqrt(beta)*np.log(emax/Eb)) - erf(0.5*(alpha-1)/np.sqrt(beta)+np.sqrt(beta)*np.log(emin/Eb)))/norm

def get_spec_exp_cutoff(gfpsm,SM,gamma,p,emax,emin):
    #outputs an array of shape (nn_gfpsm,n_SM,n_g,n_p,n_spec)
    # the Energy where the cutoff turns on is low and doesn't seem to affect anything so I'm ignoring it
    n_spec = len(emax)
    n_g = len(gamma)
    n_gfpsm = len(gfpsm)
    n_p = len(p)
    n_SM = len(SM)
    gfpsm = np.tile(gfpsm[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],(1,n_SM,n_g,n_p,n_spec))
    SM = np.tile(SM[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis],(n_gfpsm,1,n_g,n_p,n_spec))
    gamma = np.tile(gamma[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis],(n_gfpsm,n_SM,1,n_p,n_spec))
    p = np.tile(p[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis],(n_gfpsm,n_SM,n_g,1,n_spec))
    emax = np.tile(emax[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_gfpsm,n_SM,n_g,n_p,1))
    emin = np.tile(emin[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_gfpsm,n_SM,n_g,n_p,1))
    xbin =  np.sqrt(emax*emin)/p
    dx = (emax-emin)/p
    return gfpsm*SM*dx*xbin**gamma*np.exp(-xbin)


