import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import integrate

import GCE_calcs

rc('font',**{'family':'serif','serif':['Times New Roman']})
plt.rcParams.update({'font.size': 24})

n_J=400
J = np.logspace(19.,np.log10(3.e23),num=n_J)


J_prior = GCE_calcs.analysis.get_J_prior_MC(J)

J_prior_2 = GCE_calcs.analysis.get_J_prior_MC_2(J)

J_prior_3 = GCE_calcs.analysis.get_J_prior_MC_3(J)


coeff = np.array([37.5153,-1.5093,1.63e-2,3.66e-4,-2.89237e-5,5.32e-7])
h = .67
Mmw = 1.5e12
rvir = 200.
argument = np.zeros(6)
for i in range(6):
	argument[i] = coeff[i]*pow(np.log(h*Mmw),i)
conc = np.sum(argument)
print conc
print 200/conc

plt.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.96)
plt.plot(J,J_prior/J_prior.max(),'c',label = 'Zhang et al 2012', linewidth=2.0)
plt.plot(J,J_prior_2/J_prior_2.max(),'y',label = 'Pato et al 2015', linewidth=2.0)
plt.plot(J,J_prior_3/J_prior_3.max(),'m',label = 'McKee et al 2015', linewidth=2.0)
plt.xscale('log')
plt.xlim(1e20,3e23)
plt.ylim(0,1.1)
plt.xlabel(r'J-factor [GeV$^2$ cm$^{-5}$]')
plt.ylabel('Scaled Probability')
plt.legend(loc='upper left', frameon=False, fontsize=22)
#plt.title('J-factor Likelihoods')
plt.savefig('J_factor_likelihoods.pdf')
plt.clf()


r = np.logspace(0,1.5,100)
dens = GCE_calcs.analysis.density(r,30,1.1)
dens_p = GCE_calcs.analysis.density(r,30+3,1.1)
dens_m = GCE_calcs.analysis.density(r,30-3,1.1)
dens_2 = GCE_calcs.analysis.density(r,8.25,1.1)
dens_2_p = GCE_calcs.analysis.density(r,8.25+3,1.1)
dens_2_m = GCE_calcs.analysis.density(r,8.25-3,1.1)
print dens_p.shape
print r.shape

plt.plot(r, dens, 'c', label=r'R$_s$ = 30 kpc')
plt.plot(r, dens_p, 'c')
plt.plot(r, dens_m, 'c')
plt.plot(r, dens_2, 'm', label=r'R$_s$ = 8.25 kpc')
plt.plot(r, dens_2_p, 'm')
plt.plot(r, dens_2_m, 'm')
plt.fill_between(r, dens_m, dens_p, facecolor='c', alpha=0.25)
plt.fill_between(r, dens_2_m, dens_2_p, facecolor='m', alpha=0.25)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radius [kpc]')
plt.ylabel(r'$\rho / \rho_{local}$')
plt.ylim(1e-1,1e2)
plt.xlim(r[0],r[-1])
plt.title('Density Profiles with Different Scale Radii')
plt.savefig('density.png')

