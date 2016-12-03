import numpy as np
import scipy.optimize as scop
from scipy import integrate
from scipy.special import erf



def density(x,y,scale_radius,  gamma): #NFW density profile
    solar_radius = 8.25 #x is the los distance / solar radius  as to make the integral dimensionless
    R = np.sqrt(1.-2.*y*x + x*x)#R / solar radius
    return pow(R,-gamma)*pow((1 + R*solar_radius/scale_radius)/(1+solar_radius/scale_radius),gamma-3)

def J_factor(scale_radius,local_density,gamma):
    #integrating density^2 from x=0 to inf; x is line of sight distance/solarradius
    #theta = np.zeros((7,7))
    temp = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            y = np.cos(.5*(1+i+j)*np.pi/180.)
            integrand = lambda x: density(x,y,scale_radius,gamma)**2
            ans, err = integrate.quad(integrand, 0,np.inf)
            temp[i,j] = ans
    J = temp.mean()
    kpctocm = 3.08568e21
    deltaomega = (7.*np.pi/180.)**2
    return  deltaomega*J*8.25*kpctocm*local_density*local_density

def get_mu(bckgrnd,exposure,num_spec,J,sigma,mass):
    #print bckgrnd
    #should return an array of shape (n_cross,n_J,n_mass,n_spec)
    n_cross = len(sigma)
    n_mass = len(mass)
    n_J = len(J)
    exposure = np.tile(exposure,(n_mass,1))
    bckgrnd = np.tile(bckgrnd,(n_cross,n_J,n_mass,1))
    mass = np.tile(mass,(num_spec.shape[1],1)).T
    spec_over_mass = exposure*num_spec /mass**2
    #magic = [num_spec,J,sigma,inv_mass_sqrd]
    #I've no idea why the following line works, but it apparently calculates the outer product of the multiple vectors in 'magic'
    #residual = reduce(np.multiply, np.ix_(*magic))/(8*np.pi)
    #found a more reasonable way to do the above
    residual = np.einsum('i,j,kl->ijkl',sigma,J,spec_over_mass)/(8.*np.pi)
    #print residual[0,0,25,:]
    return bckgrnd + residual

def conc():
    coeff = np.array([37.5153,-1.5093,1.63e-2,3.66e-4,-2.89237e-5,5.32e-7])
    h = 0.67
    Mmw = 1.5e12
    rvir = 200.
    conc = 0.
    for i in range(6):
        conc += coeff[i]*np.log(h*Mmw)**i
    return conc

def get_mu_log_parab(bckgrnd,exposure,num_spec):
    #each of the inputs should be arrays of the shape (n_N0, n_alpha, n_beta, n_eb, n_spec)
    return bckgrnd + exposure*num_spec

def get_spec_log_parab(N0,alpha,beta,Eb,emax,emin):
    #emax is a vector of len n_spec, the maxima energy of each bin
    #emin is a vector of len n_spec, the minima energy of each bin
    #E_b is a vector of len n_eb, the scale energy for the log-parabola
    #beta is a vector of len n_beta, the parameter for exponential cutoff,
    #alpha is a vector of len n_alpha, the parameter for the power law part of the spectra
    #N0 is a vector of len n_N0, the parameter for the normalization
    #print N0
    #print alpha
    #print beta
    #print Eb
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
    return N0*Eb*0.5*np.sqrt(np.pi/beta) * np.exp(-0.25*(alpha-1)**2/beta) * (erf(0.5*(alpha-1)/np.sqrt(beta) + np.sqrt(beta)*np.log(emax/Eb)) - erf(0.5*(alpha-1)/np.sqrt(beta) + np.sqrt(beta)*np.log(emin/Eb)))


def get_dn_de_log_parab(N0,alpha,beta,Eb,energy):
    n_spec = len(energy)
    n_eb = len(Eb)
    n_beta = len(beta)
    n_alpha = len(alpha)
    n_N0 = len(N0)
    N0 = np.tile(N0[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],(1,n_alpha,n_beta,n_eb,n_spec))
    alpha = np.tile(alpha[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis],(n_N0,1,n_beta,n_eb,n_spec))
    beta = np.tile(beta[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis],(n_N0,n_alpha,1,n_eb,n_spec))
    Eb = np.tile(Eb[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis],(n_N0,n_alpha,n_beta,1,n_spec))
    energy = np.tile(energy[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
    return N0*(energy/Eb)**(-alpha - beta*np.log(energy/Eb))
