import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate

import GCE_calcs



#channel = 0  #tau
channel = 1 #bbar

#model = 0  #MG aka full
#model = 1  #noMG
model = 2  #IC data

###################
######  GCE part
###################

trunc = 8  #how many data points truncated from the front
dataset=20 #largest data point

mass_table = np.array([np.loadtxt('spectra/unbinned/tau/tau_mass_table.txt')[:,1],
                       np.loadtxt('spectra/unbinned/bbar/bbar_mass_table.txt')[:,1]])[channel] #table of masses

#mass_table = np.array([])[channel]

file_path = np.array(['spectra/binned/tau/binned_spectra_','spectra/binned/bbar/binned_spectra_'])[channel]

file_name = np.array([['tau_full','tau_noMG','tau_IC'],['bbar_full','bbar_noMG','bbar_IC']])[channel][model]

n_J=400
J = np.logspace(19.,26.,num=n_J)

n_mass = len(mass_table)

n_sigma = 100
sigma = np.logspace(-30.,-23.,num=n_sigma)

sigma_prior_norm = np.trapz(np.ones(n_sigma),x = np.log(sigma))# a flat prior in linear space for mass and logspace for cross section

mass_prior_norm = np.trapz(np.ones(n_mass),x = mass_table)

raw= np.array([np.loadtxt('data/background/GCE_paper_fullmodel_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_noMG_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_IC_spectrum.dat')])[model]
               
bin_center = raw[trunc:dataset,0]#logarthmic bin center
k = raw[trunc:dataset,5]
background = raw[trunc:dataset,7]
exposure = raw[trunc:dataset,6]

binned_spectra = np.loadtxt(file_path+file_name+'_'+str(dataset)+'_'+str(trunc)+'_GCE.txt')

mu = GCE_calcs.calculations.get_mu(background,exposure, binned_spectra, J, sigma, mass_table)

k = np.tile(k,(n_sigma,n_J,n_mass,1))

log_like_4d = GCE_calcs.analysis.poisson_log_like(k,mu) #a 4-d array of the log-likelihood with shape (n_sigma,n_J,n_mass,n_spec)

log_like_3d = np.sum(log_like_4d,axis=3) #summing the log-like along the energy bin axis

J_prior = GCE_calcs.analysis.get_J_prior_MC(J)

norm_test = np.trapz(J_prior,x=J)
assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the J-factor is off by more than 1%'

J_prior = np.tile(J_prior[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))

GCE_like_3d = np.exp(log_like_3d)*J_prior

max_index_GCE = np.unravel_index(GCE_like_3d.argmax(),GCE_like_3d.shape)
plt.plot(10**bin_center,background, label = 'background')
plt.errorbar(10**bin_center,k[0,0,0,:],yerr = np.sqrt(k[0,0,0,:]),label = 'observed counts')
plt.plot(10**bin_center,mu[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2],:],label = 'expected number counts')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.legend(loc='best')
plt.savefig('plots/WIMP/test_residuals_GCE.png')
plt.clf()

GCE_like_2d = np.trapz(GCE_like_3d,x=J,axis=1)

levels = [0,1,3,6,10,15,25,36,49,100]
CS = plt.contour(mass_table,sigma,-np.log(GCE_like_2d) + np.log(GCE_like_2d.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.yscale('log')
plt.xlabel('Mass [GeV]')
plt.ylabel('Cross Section [cm^3 sec^-1]')
plt.savefig('plots/WIMP/GCE_contours.png')
plt.clf()

evidence_GCE = np.trapz(np.trapz(GCE_like_2d,x = np.log(sigma),axis =0),x = mass_table,axis=0) / (sigma_prior_norm * mass_prior_norm)

################
### end GCE part
################


###################
### Dwarfs
###################


like_name = np.array(['like_bootes_I',
                        'like_canes_venatici_I',
                        'like_canes_venatici_II',
                        'like_carina',
                        'like_coma_berenices',
                        'like_draco',
                        'like_fornax',
                        'like_hercules',
                        'like_leo_I',
                        'like_leo_II',
                        'like_leo_IV',
                        'like_sculptor',
                        'like_segue_1',
                        'like_sextans',
                        'like_ursa_major_I',
                        'like_ursa_major_II',
                        'like_ursa_minor',
                        'like_willman_1'])


dwarf_mean_J = np.array([18.8,
                        17.7,
                        17.9,
                        18.1,
                        19.0,
                        18.8,
                        18.2,
                        18.1,
                        17.7,
                        17.6,
                        17.9,
                        18.6,
                        19.5,
                        18.4,
                        18.3,
                        19.3,
                        18.8,
                        19.1])

dwarf_var_J = np.array([0.22,
                        0.26,
                        0.25,
                        0.23,
                        0.25,
                        0.16,
                        0.21,
                        0.25,
                        0.18,
                        0.18,
                        0.28,
                        0.18,
                        0.29,
                        0.27,
                        0.24,
                        0.28,
                        0.19,
                        0.31])


binned_energy_spectra_dwarf =  np.loadtxt(file_path+file_name+'_'+str(dataset)+'_'+str(trunc)+'dwarf_energy_spectra.txt')

like_dwarf_2d = np.ones((n_sigma,n_mass))
for i in range(len(like_name)):
    name = like_name[i]
    print name
    J_dwarf = np.logspace(dwarf_mean_J[i] - 5*dwarf_var_J[i],dwarf_mean_J[i]+5*dwarf_var_J[i],n_J)
    J_prior_dwarf = GCE_calcs.analysis.get_J_prior_dwarf(J_dwarf,dwarf_mean_J[i],dwarf_var_J[i])
    norm_test = np.trapz(J_prior_dwarf, x=J_dwarf)
    assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the J-factor is off by more than 1%'
    J_prior_dwarf = np.tile(J_prior_dwarf[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))
    espec_dwarf = GCE_calcs.calculations.get_eflux(binned_energy_spectra_dwarf,J_dwarf,sigma,mass_table)
    log_like_dwarf_4d = GCE_calcs.analysis.dwarf_delta_log_like(espec_dwarf,name)
    log_like_dwarf_3d = np.sum(log_like_dwarf_4d,axis=3)
    like_dwarf_3d = np.exp(log_like_dwarf_3d)*J_prior_dwarf
    like_ind_2d = np.trapz(like_dwarf_3d,x=J_dwarf,axis=1)
    CS = plt.contour(mass_table,sigma,-np.log(like_ind_2d) + np.log(like_ind_2d.max()),levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.yscale('log')
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Cross Section [cm^3 sec^-1]')
    plt.savefig('plots/WIMP/'+name+'_contours.png')
    plt.clf()
    like_dwarf_2d *= like_ind_2d


CS = plt.contour(mass_table,sigma,-np.log(like_dwarf_2d/like_dwarf_2d.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.yscale('log')
plt.xlabel('Mass [GeV]')
plt.ylabel('Cross Section [cm^3 sec^-1]')
plt.savefig('plots/WIMP/dwarf_contours.png')
plt.clf()


evidence_dwarf = np.trapz(np.trapz(like_dwarf_2d ,x = np.log(sigma),axis=0),x=mass_table,axis=0)/ (sigma_prior_norm * mass_prior_norm)

print 'the dwarf evidence is ' +str(evidence_dwarf)
print 'the GCE evidence is ' + str(evidence_GCE)
print 'the product of the dwarf and GCE evidence is ' + str(evidence_dwarf*evidence_GCE) 

####################
####### C-C-C-COMBO
####################

combo_like_2d = like_dwarf_2d  * GCE_like_2d
evidence_combo = np.trapz(np.trapz(combo_like_2d ,x = np.log(sigma),axis=0),x =mass_table,axis=0)/ (sigma_prior_norm * mass_prior_norm)

print 'the combined evidence is ' +str(evidence_combo)
