import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm


def density(theta,z,scale_radius,  gamma): #NFW density profile divided by local density, hence unitless
    #theta is a vector with length n_theta in radians, theta is the polar angle where the z-axis is through the center of the galaxy, not galactic longitude or latitude
    #z is a vector with length n_z, it is the line of sight distance from the local position, it has units kpc, gets divided by solar radius to make it unitless
    #scale_radius and gamma are scalars
    solar_radius = 8.25
    n_theta = len(theta)
    n_z = len(z)
    theta = np.tile(theta[:,np.newaxis],(1,n_z))
    z = np.tile(z[np.newaxis,:],(n_theta,1)) / solar_radius
    R = np.sqrt(1 -2.*np.cos(theta)*z + z*z)#R / solar radius
    return pow(R,-gamma)*pow((1 + R*solar_radius/scale_radius)/(1+solar_radius/scale_radius),gamma-3)


def J_integral(scale_radius,gamma):
    #calculates the J_factor over a 3.5x3.5 region
    theta = np.linspace(0.001,3.5*np.pi/180.,200)# polar angle from z
    z = 8.25*np.linspace(0.5,1.5,200)#line of sight distance in kpc
    kpctocm = 3.086e21
    return np.trapz(2.*np.pi*np.sin(theta)* np.trapz(density(theta,z,scale_radius,gamma)**2, x=z, axis=1), x=theta, axis=0)*kpctocm




coeff = np.array([37.5153,-1.5093,1.63e-2,3.66e-4,-2.89237e-5,5.32e-7])
h = .67
Mmw = 1.5e12
rvir = 200.
argument = np.zeros(6)
for i in range(6):
	argument[i] = coeff[i]*pow(np.log(h*Mmw),i)
conc = np.sum(argument)

n_MC = 1000000

conc_array = np.random.lognormal(np.log(conc),.14,n_MC)
gamma_array = np.random.normal(1.12,.05,n_MC)
rho = np.random.normal(0.28,0.08,n_MC)

J_array = np.zeros(n_MC)

print 'doing the MC'
for i in range(n_MC):
    J_array[i] = rho[i]**2*J_integral(rvir/conc_array[i],gamma_array[i])

print 'doing the KDE'
J_kde = stats.gaussian_kde(J_array)

print 'plotting'
n_J=41
J_range = np.logspace(19., 23., num=n_J, endpoint=True)
#J_range = np.linspace(0,5,40)*1.e23
plt.plot(J_range,J_kde(J_range))
plt.xscale('log')
plt.savefig('J_MC2.png')
plt.clf()

np.savetxt('J_MC.txt',(J_range,J_kde(J_range)))

