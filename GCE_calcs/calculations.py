import numpy as np
import scipy.optimize as scop
from scipy import integrate


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

def mu(bckgrnd,num_spec,J,log_sigma,mass):
    return bckgrnd + num_spec*J*pow(10,log_sigma)/(8*np.pi*mass**2)

def conc():
    coeff = np.array([37.5153,-1.5093,1.63e-2,3.66e-4,-2.89237e-5,5.32e-7])
    h = 0.67
    Mmw = 1.5e12
    rvir = 200.
    conc = 0.
    for i in range(6):
        conc += coeff[i]*np.log(h*Mmw)**i
    return conc

