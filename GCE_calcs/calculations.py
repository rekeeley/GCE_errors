import numpy as np
import scipy.optimize as scop
from scipy import integrate
from scipy.special import erf, gammainc



def density(theta,z,scale_radius,  gamma): #NFW density profile
    #theta is a vector with length n_theta in radians, theta is the polar angle where the z-axis is throught the center of the galaxy, not galactic longitude or latitude
    #z is a vector with length n_z
    #scale_radius and gamma are scalars
    solar_radius = 8.25
    n_theta = len(theta)
    n_z = len(z)
    theta = np.tile(theta[:,np.newaxis],(1,n_z))
    z = np.tile(z[np.newaxis,:],(n_theta,1))
    R = np.sqrt(1 -2.*np.cos(theta)*z + z*z)#R / solar radius
    return pow(R,-gamma)*pow((1 + R*solar_radius/scale_radius)/(1+solar_radius/scale_radius),gamma-3)

def J_integral(scale_radius,gamma):
    theta = np.linspace(0,7*np.pi/180.,200)
    z = np.linspace(0,400,200)
    return np.trapz(2.*np.pi*np.sin(theta)* np.trapz(density(theta,z,scale_radius,gamma)**2, x=z, axis=1), x=theta, axis=0)


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


def get_mu_log_parab(background,exposure,num_spec):
    #each of the inputs should be arrays of the shape (n_N0, n_alpha, n_beta, n_eb, n_spec)
    return background + exposure*num_spec

def get_spec_log_parab(N0,alpha,beta,Eb,emax,emin):
    #emax is a vector of len n_spec, the maxima energy of each bin
    #emin is a vector of len n_spec, the minima energy of each bin
    #E_b is a vector of len n_eb, the scale energy for the log-parabola
    #beta is a vector of len n_beta, the parameter for exponential cutoff,
    #alpha is a vector of len n_alpha, the parameter for the power law part of the spectra
    #N0 is a vector of len n_N0, the parameter for the normalization
    n_spec = len(emax)
    n_eb = len(Eb)
    n_beta = len(beta)
    n_alpha = len(alpha)
    n_N0 = len(N0)
    N0 = np.tile(N0[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],(1,n_alpha,n_beta,n_eb,n_spec))
    alpha = np.tile(alpha[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis],(n_N0,1,n_beta,n_eb,n_spec))
    beta = np.tile(beta[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis],(n_N0,n_alpha,1,n_eb,n_spec))
    Eb = np.tile(Eb[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis],(n_N0,n_alpha,n_beta,1,n_spec))
    emax = np.tile(emax[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
    emin = np.tile(emin[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
    #print erf( ((alpha-1) + 2*beta*np.log(emax/Eb)) / (2*beta**0.5) )
    #return N0 * Eb*np.exp(- (alpha-1)**2 / (4*beta))*np.pi**0.5 * (erf( ((alpha-1) + 2*beta*np.log(emax/Eb)) / (2*beta**0.5) ) - erf( ((alpha-1) + 2*beta*np.log(emin/Eb)) / (2*beta**0.5))) / (2*beta**0.5)
    #return N0 * erf( ((alpha-1) + 2*beta*np.log(emax/Eb)) / (2*beta**0.5) )
    #return N0 *  ((alpha-1) + 2*beta*np.log(emax/Eb)) / (2*beta**0.5)
    #return N0*Eb*0.5*np.sqrt(np.pi/beta) * np.exp(-0.25*(alpha-1)**2/beta) * (erf(0.5*(alpha-1)/np.sqrt(beta) + np.sqrt(beta)*np.log(emax/Eb)) - erf(0.5*(alpha-1)/np.sqrt(beta) + np.sqrt(beta)*np.log(emin/Eb)))
    return N0 * (erf(0.5*(alpha-1)/np.sqrt(beta)+np.sqrt(beta)*np.log(emax/Eb)) - erf(0.5*(alpha-1)/np.sqrt(beta)+np.sqrt(beta)*np.log(emin/Eb)))



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


def get_spec_exp_cutoff(N0,gamma,p,emax,emin):
    #outputs an array of shape (n_N0,n_g,n_p,n_spec)
    # the Energy where the cutoff turns on is low and doesn't seem to affect anything so I'm ignoring it
    n_spec = len(emax)
    n_g = len(gamma)
    n_N0 = len(N0)
    n_p = len(p)
    E_s = 0.2
    N0 = np.tile(N0[:,np.newaxis,np.newaxis,np.newaxis],(1,n_g,n_p,n_spec))
    gamma = np.tile(gamma[np.newaxis,:,np.newaxis,np.newaxis],(n_N0,1,n_p,n_spec))
    p = np.tile(p[np.newaxis,np.newaxis,:,np.newaxis],(n_N0,n_g,1,n_spec))
    emax = np.tile(emax[np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_g,n_p,1))
    emin = np.tile(emin[np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_g,n_p,1))
    #return N0* E_s**-gamma * p**(1+gamma) * (gammainc(1+gamma,emax/p) - gammainc(1+gamma,emin/p))
    return N0 * (gammainc(1+gamma,emax/p) - gammainc(1+gamma,emin/p))



