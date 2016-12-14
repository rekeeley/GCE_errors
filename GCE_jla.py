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

like_name = np.array(['like_bootes_I',
                        'like_bootes_II',
                        'like_bootes_III',
                        'like_canes_venatici_I',
                        'like_canes_venatici_II',
                        'like_canis_major',
                        'like_carina',
                        'like_coma_berenices',
                        'like_draco',
                        'like_fornax',
                        'like_hercules',
                        'like_leo_I',
                        'like_leo_II',
                        'like_leo_IV',
                        'like_leo_V',
                        'like_pisces_II',
                        'like_sagittarius',
                        'like_sculptor',
                        'like_segue_1',
                        'like_segue_2',
                        'like_sextans',
                        'like_ursa_major_I',
                        'like_ursa_major_II',
                        'like_ursa_minor',
                        'like_willman_1'])



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


N0_dwarf = np.logspace(-8,-5,100)
N0_dwarf_normalization = np.trapz(np.ones(len(N0_dwarf)),x = np.log(N0_dwarf))
log_like_dwarf_2d = np.zeros((n_alpha,n_beta))
for name in like_name:
    data_energy_dwarf = np.loadtxt('release-01-00-02/'+name+'.txt')
    emin_dwarf = np.unique(data_energy_dwarf[:,0])/1000.
    emax_dwarf = np.unique(data_energy_dwarf[:,1])/1000. #delete copies and convert from MeV to GeV
    data_dwarf = np.loadtxt('dwarf_re_data/'+name+'_data.txt')
    k_dwarf = data_dwarf[:,0]
    back_flux_dwarf = data_dwarf[:,1]
    expo_dwarf = data_dwarf[:,2]
    binned_spectra_dwarf = GCE_calcs.calculations.get_spec_log_parab(N0_dwarf,alpha,beta,Eb,emax_dwarf,emin_dwarf)
    mu_dwarf = GCE_calcs.calculations.get_mu(expo_dwarf*back_flux_dwarf,expo_dwarf,binned_spectra_dwarf)
    log_like_dwarf_5d = GCE_calcs.analysis.poisson_log_like(k_dwarf,mu_dwarf)
    log_like_dwarf_4d = np.sum(log_like_dwarf_5d,axis=4)
    print log_like_draco_4d.max()
    log_like_dwarf_2d += np.log(np.trapz(np.exp(log_like_draco_4d[:,:,:,0]) - log_like_draco_4d.max(),x=np.log(N0_dwarf),axis=0)/N0_dwarf_normalization)

like_dwarf =  np.trapz(np.trapz(np.exp(log_like_dwarf_2d - log_like_dwarf_2d.max()),x = alpha,axis=0),x=beta,axis=0)

print log_like_dwarf_2d.max()
print alpha_prior_norm
print beta_prior_norm
print like_dwarf



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

