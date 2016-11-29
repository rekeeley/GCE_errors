import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate

import GCE_calcs



#channel = 0  #tau
channel = 1 #bbar

model = 0  #MG aka full
#model = 1  #noMG
#model = 2  IC data

###################
######  GCE part
###################

trunc = 8  #how many data points truncated from the front
dataset=20 #largest data point

mass_table = np.array([np.loadtxt('spectra/tau/LSP-energies-original.dat')[:,1],
                       np.loadtxt('spectra/bbar/LSP-energies.dat')[:,1]])[channel] #table of masses

#mass_table = np.array([])[channel]

file_path = np.array(['spectra/tau/output-gammayield-','spectra/bbar/muomega-gammayield-'])[channel]

output_file = np.array([['tau_full','tau_noMG','tau_IC'],['bbar_full','bbar_noMG','bbar_IC']])[channel][model]

raw= np.array([np.loadtxt('data/background/GCE_paper_fullmodel_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_noMG_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_IC_spectrum.dat')])[model]
               
bin_center = raw[trunc:dataset,0]#logarthmic bin center

k = raw[trunc:dataset,5]

background = raw[trunc:dataset,7]

exposure = raw[trunc:dataset,6]

binned_spectra = np.loadtxt('spectra/binned/binned_spectra_'+output_file+'_'+str(dataset)+'_'+str(trunc)+'.dat')
#binned_spectra = np.loadtxt('spectra/test/binned/binned_spectra_'+output_file+'_'+str(dataset)+'_'+str(trunc)+'.dat')

n_J=400
J = np.logspace(20.,24.,num=n_J)

n_mass = len(mass_table)

n_sigma = 40
sigma = np.logspace(-27.,-24.,num=n_sigma)

sigma_mass_prior = 1. / (n_sigma*n_mass)# a flat prior in linear space for mass and logspace for cross section

mu = GCE_calcs.calculations.get_mu(background,exposure, binned_spectra, J, sigma, mass_table)

k = np.tile(k,(n_sigma,n_J,n_mass,1))

log_like_4d = GCE_calcs.analysis.poisson_log_like(k,mu) #a 4-d array of the log-likelihood with shape (n_sigma,n_J,n_mass,n_spec)

log_like_3d = np.sum(log_like_4d,axis=3) #summing the log-like along the energy bin axis

J_prior = GCE_calcs.analysis.get_J_prior_MC(J)

norm_test = np.trapz(J_prior,x=J)
assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the J-factor is off by more than 1%'

J_prior = np.tile(J_prior[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))

GCE_like_3d = np.exp(log_like_3d)*J_prior

GCE_like_2d = np.trapz(GCE_like_3d,x=J,axis=1)

evidence_GCE = np.sum(GCE_like_2d)*sigma_mass_prior

print evidence_GCE
################
### end GCE part
################


###################
### Dwarfs
###################


data_draco = np.loadtxt('dwarf_re_data/like_draco_data.txt')

binned_spectra_draco = np.loadtxt('spectra/test2/binned/binned_spectra_bbar_IC_draco.txt')

k_draco = data_draco[:,0]

b_flux_draco = data_draco[:,1]

exp_draco = data_draco[:,2]

J_draco = np.logspace(18,20,n_J)

J_draco_prior = GCE_calcs.analysis.get_J_prior_Draco(J_draco)
norm_test = np.trapz(J_draco_prior,x=J_draco)
assert abs(norm_test - 1) < 0.01, 'draco prior not normalized'

J_draco_prior = np.tile(J_draco_prior[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))

mu_draco = GCE_calcs.calculations.get_mu(b_flux_draco*exp_draco, exp_draco, binned_spectra_draco, J_draco, sigma, mass_table)

k_draco = np.tile(k_draco,(n_sigma,n_J,n_mass,1))

log_like_draco_4d = GCE_calcs.analysis.poisson_log_like(k_draco,mu_draco)

log_like_draco_3d = np.sum(log_like_draco_4d,axis=3)

draco_like_3d = np.exp(log_like_draco_3d)*J_draco_prior

draco_like_2d = np.trapz(draco_like_3d,x=J_draco,axis=1)

evidence_draco = np.sum(draco_like_2d)*sigma_mass_prior

print evidence_draco




#levels = [0,1,3,6,12]
#plt.contour(mass_table,log_sigma,-post_log_pdf[:,2,:],levels)
#plt.contour(mass_table,np.log10(sigma),-post_nlp_min2,levels)
#plt.contour(mass_table,np.log10(sigma),-post_test4,levels)
#plt.savefig('array_test.png')



#plt.plot(bin_center,k,label = 'data')
#plt.plot(bin_center,background,label = 'background')
#for i in range(5):
#    plt.plot(bin_center,mu[n_sigma/2,n_J/2,5+10*i,:],label = str(mass_table[5+10*i]))
#plt.legend(loc='best')
#plt.savefig('spectra_test.png')
#plt.clf()

