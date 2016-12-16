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
J = np.logspace(20.,24.,num=n_J)

n_mass = len(mass_table)

n_sigma = 100
sigma = np.logspace(-27.,-22.,num=n_sigma)

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

GCE_like_2d = np.trapz(GCE_like_3d,x=J,axis=1)

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


binned_spectra_dwarf = np.loadtxt(file_path+file_name+'_'+str(dataset)+'_'+str(trunc)+'_dwarf.txt')

log_like_dwarf_2d = np.zeros((n_sigma,n_mass))
dwarf_log_factor = 0.0
for i in range(len(like_name)):
    name = like_name[i]
    data_energy_dwarf = np.loadtxt('release-01-00-02/'+name+'.txt')
    emin_dwarf = np.unique(data_energy_dwarf[:,0])/1000.
    emax_dwarf = np.unique(data_energy_dwarf[:,1])/1000. #delete copies and convert from MeV to GeV
    data_dwarf = np.loadtxt('dwarf_re_data/'+name+'_data.txt')
    k_dwarf = data_dwarf[:,0]
    k_dwarf = np.tile(k_dwarf[np.newaxis,np.newaxis,np.newaxis,:],(n_sigma,n_J,n_mass,1))
    back_flux_dwarf = data_dwarf[:,1]
    expo_dwarf = data_dwarf[:,2]
    J_dwarf = np.logspace(dwarf_mean_J[i] - 5*dwarf_var_J[i],dwarf_mean_J[i]+5*dwarf_var_J[i],n_J)
    J_prior_dwarf = GCE_calcs.analysis.get_J_prior_dwarf(J_dwarf,dwarf_mean_J[i],dwarf_var_J[i])
    norm_test = np.trapz(J_prior_dwarf, x=J_dwarf)
    assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the J-factor is off by more than 1%'
    J_prior_dwarf = np.tile(J_prior_dwarf[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))
    mu_dwarf = GCE_calcs.calculations.get_mu(expo_dwarf*back_flux_dwarf, expo_dwarf, binned_spectra_dwarf, J_dwarf, sigma, mass_table)
    log_like_dwarf_4d = GCE_calcs.analysis.poisson_log_like(k_dwarf,mu_dwarf)
    log_like_dwarf_3d = np.sum(log_like_dwarf_4d,axis=3)
    log_like_dwarf_3d += np.log(J_prior_dwarf)
    max_index_dwarf = np.unravel_index(log_like_dwarf_3d.argmax(),log_like_dwarf_3d.shape)
    print 'the index of the max prob is ' + str(max_index_dwarf)
    plt.plot(0.5*(emin_dwarf + emax_dwarf),back_flux_dwarf*expo_dwarf, label = 'background')
    plt.errorbar(0.5*(emin_dwarf + emax_dwarf),k_dwarf[0,0,0,:],yerr = np.sqrt(k_dwarf[0,0,0,:]),label = 'observed counts')
    plt.plot(0.5*(emin_dwarf + emax_dwarf),mu_dwarf[max_index_dwarf[0],max_index_dwarf[1],max_index_dwarf[2],:],label = 'expected number counts')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number Counts')
    plt.legend(loc='best')
    plt.savefig('test_residuals/dwarf_'+str(i)+'.png')
    plt.clf()
    print log_like_dwarf_3d.max()
    dwarf_log_factor +=log_like_dwarf_3d.max()
    log_like_dwarf_2d += np.log(np.trapz(np.exp(log_like_dwarf_3d) - log_like_dwarf_3d.max(),x=J_dwarf,axis=1))

like_dwarf =  np.trapz(np.trapz(np.exp(log_like_dwarf_2d - log_like_dwarf_2d.max()),x = np.log(sigma),axis=0),x=mass_table,axis=0)

#print 'the dwarf log factor is ' + str(dwarf_log_factor)
print 'the dwarf evidence is ' +str(like_dwarf) + ' times e to the ' +str(log_like_dwarf_2d.max()) +' times e to the ' +str(dwarf_log_factor)

print 'the GCE evidence is ' + str(evidence_GCE)

print 'the product of the dwarf and GCE evidence is ' + str(like_dwarf*evidence_GCE) + ' times e to the '+str(dwarf_log_factor)

####################
####### C-C-C-COMBO
####################

combo_log_like_2d = log_like_dwarf_2d + np.log(GCE_like_2d)
combo_like = np.trapz(np.trapz(np.exp(log_like_dwarf_2d - log_like_dwarf_2d.max())*GCE_like_2d,x = np.log(sigma),axis=0),x =mass_table,axis=0)

print combo_like
print 'the combined evidence is ' +str(combo_like) + ' times e to the ' + str(dwarf_log_factor) + ' times e to the ' + str(log_like_dwarf_2d.max())