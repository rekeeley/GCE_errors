import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate

import GCE_calcs

############################
# Log-Parabola Model (normalization for GCE / dwarfs are independent)
############################

###################
######  GCE part
###################

#channel = 0  #tau
channel = 1 #bbar

model = 0  #MG aka full
#model = 1  #noMG
#model = 2  IC data

trunc = 8  #how many data points truncated from the front
dataset=20 #largest data point

raw= np.array([np.loadtxt('data/background/GCE_paper_fullmodel_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_noMG_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_IC_spectrum.dat')])[model]
               
bin_center = raw[trunc:dataset,0]#logarthmic bin center
log_bin_width = bin_center[1] - bin_center[0]
emin_GCE = 10**(bin_center - log_bin_width)
emax_GCE = 10**(bin_center + log_bin_width)
k = raw[trunc:dataset,5]
background = raw[trunc:dataset,7]
exposure = raw[trunc:dataset,6]

Eb = np.array([0.2])
alpha = np.linspace(-3,1.5,100)
beta = np.linspace(0.01,1.5,100)
N0_GCE = 3.*np.logspace(-8,-5,100)

n_eb = len(Eb)
n_beta = len(beta)
n_alpha = len(alpha)
n_N0 = len(N0_GCE)

beta_prior_norm = np.trapz(np.ones(n_beta), x = beta)
alpha_prior_norm = np.trapz(np.ones(n_alpha),x = alpha)
N0_prior_norm = np.trapz(np.ones(n_N0),x = np.log(N0_GCE))

binned_spectra = GCE_calcs.calculations.get_spec_log_parab(N0_GCE,alpha,beta,Eb,emax_GCE,emin_GCE)

n_eb = len(Eb)
n_beta = len(beta)
n_alpha = len(alpha)
n_N0 = len(N0_GCE)

background = np.tile(background[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
exposure = np.tile(exposure[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
k = np.tile(k[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))

mu_GCE = GCE_calcs.calculations.get_mu_log_parab(background,exposure,binned_spectra)

log_like_GCE_5d = GCE_calcs.analysis.poisson_log_like(k,mu_GCE)

log_like_GCE_4d = np.sum(log_like_GCE_5d,axis=4)

GCE_log_factor = log_like_GCE_4d.max()

log_like_GCE_2d = np.trapz(np.exp(log_like_GCE_4d[:,:,:,0]-log_like_GCE_4d.max()),x = np.log(N0_GCE),axis = 0)/N0_prior_norm

like_GCE = np.trapz(np.trapz(np.exp(log_like_GCE_2d),x = alpha,axis=0),x=beta,axis=0)

print GCE_log_factor
print like_GCE

###################
###### dwarfs part
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



N0_dwarf = np.logspace(-8,-5,100)
n_N0_dwarf  = len(N0_dwarf)
N0_dwarf_normalization = np.trapz(np.ones(len(N0_dwarf)),x = np.log(N0_dwarf))
dwarf_log_factor = 0
log_like_dwarf_2d = np.zeros((n_alpha,n_beta))
for name in like_name:
    data_energy_dwarf = np.loadtxt('release-01-00-02/'+name+'.txt')
    emin_dwarf = np.unique(data_energy_dwarf[:,0])/1000.
    emax_dwarf = np.unique(data_energy_dwarf[:,1])/1000. #delete copies and convert from MeV to GeV
    data_dwarf = np.loadtxt('dwarf_re_data/'+name+'_data.txt')
    k_dwarf = data_dwarf[:,0]
    k_dwarf = np.tile(k_dwarf[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0_dwarf,n_alpha,n_beta,n_eb,1))
    back_flux_dwarf = data_dwarf[:,1]
    back_flux_dwarf = np.tile(back_flux_dwarf[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0_dwarf,n_alpha,n_beta,n_eb,1))
    expo_dwarf = data_dwarf[:,2]
    expo_dwarf = np.tile(expo_dwarf[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0_dwarf,n_alpha,n_beta,n_eb,1))
    binned_spectra_dwarf = GCE_calcs.calculations.get_spec_log_parab(N0_dwarf,alpha,beta,Eb,emax_dwarf,emin_dwarf)
    mu_dwarf = GCE_calcs.calculations.get_mu_log_parab(expo_dwarf*back_flux_dwarf,expo_dwarf,binned_spectra_dwarf)
    log_like_dwarf_5d = GCE_calcs.analysis.poisson_log_like(k_dwarf,mu_dwarf)
    log_like_dwarf_4d = np.sum(log_like_dwarf_5d,axis=4)
    #print log_like_dwarf_4d.max()
    dwarf_log_factor += log_like_dwarf_4d.max()
    log_like_dwarf_2d += np.log(np.trapz(np.exp(log_like_dwarf_4d[:,:,:,0]) - log_like_dwarf_4d.max(),x=np.log(N0_dwarf),axis=0)/N0_dwarf_normalization)

like_dwarf =  np.trapz(np.trapz(np.exp(log_like_dwarf_2d ),x = alpha,axis=0),x=beta,axis=0)


print 'the evidence is '+str(like_dwarf) + ' times e to the ' + str(dwarf_log_factor)

####################
####### C-C-C-COMBO
####################

#combo_like_2d = like_GCE_2d*alpha_beta_posterior / N0_draco_normalization

combo_log_like_2d = log_like_dwarf_2d + log_like_GCE_2d


#levels = [0,1,3,6,10,15,25,36,49,100]
#plt.contour(alpha,beta,-np.log(combo_like_2d.T  / combo_like_2d.max()),levels)
#plt.xlabel('alpha')
#plt.ylabel('beta')
#plt.savefig('combo_alpha_beta_posterior.png')
#plt.clf()

combo_like = np.trapz(np.trapz(np.exp(combo_log_like_2d),x = alpha, axis = 0),x=beta,axis=0)

print dwarf_log_factor + GCE_log_factor
print combo_like

