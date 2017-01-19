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


##### DIFFERENT MODELS FOR THE BACKGOUND #####
#model = 0  #MG aka full
#model = 1  #noMG
model = 2  #IC data

trunc = 8  #how many data points truncated from the front
dataset=20 #largest data point

raw= np.array([np.loadtxt('data/background/GCE_paper_fullmodel_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_noMG_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_IC_spectrum.dat')])[model]

file_name = np.array(['log_parab_full','log_parab_noMG','log_parab_IC'])[model]
               
bin_center = raw[trunc:dataset,0]#logarthmic bin center
log_bin_width = bin_center[1] - bin_center[0]
emin_GCE = 10**(bin_center - log_bin_width)
emax_GCE = 10**(bin_center + log_bin_width)
k = raw[trunc:dataset,5]
background = raw[trunc:dataset,7]
exposure = raw[trunc:dataset,6]

Eb = np.array([0.2])
alpha = np.linspace(-5.5,0,100)
beta = np.linspace(0.4,1.7,100)
N0_GCE = np.logspace(-8,-3,150)

n_eb = len(Eb)
n_beta = len(beta)
n_alpha = len(alpha)
n_N0 = len(N0_GCE)

beta_prior_norm = np.trapz(np.ones(n_beta), x = beta)
alpha_prior_norm = np.trapz(np.ones(n_alpha),x = alpha)
N0_prior_norm = np.trapz(np.ones(n_N0),x = np.log(N0_GCE))

binned_spectra = GCE_calcs.calculations.get_spec_log_parab(N0_GCE,alpha,beta,Eb,emax_GCE,emin_GCE)
# 5-d array of shape n_N0, n_alpha, n_beta, n_Eb, n_spec.  Is the number flux


background = np.tile(background[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
exposure = np.tile(exposure[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
k = np.tile(k[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))

mu_GCE = GCE_calcs.calculations.get_mu_log_parab(background,exposure,binned_spectra)

log_like_GCE_5d = GCE_calcs.analysis.poisson_log_like(k,mu_GCE)

log_like_GCE_4d = np.sum(log_like_GCE_5d,axis=4)

max_index_GCE = np.unravel_index(log_like_GCE_4d.argmax(),log_like_GCE_4d.shape)
plt.errorbar(10**bin_center,k[0,0,0,0,:]-background[0,0,0,0,:],yerr = np.sqrt(k[0,0,0,0,:]),label = 'Observed Residual')
plt.plot(10**bin_center,mu_GCE[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2],max_index_GCE[3],:]-background[0,0,0,0,:],label = 'Expected Residual')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.legend(loc='best')
plt.savefig('plots/log_parabola/'+file_name+'test_residuals_GCE.png')
plt.clf()

GCE_like_2d = np.trapz(np.exp(log_like_GCE_4d[:,:,:,0]), x=np.log(N0_GCE), axis=0)/N0_prior_norm

levels = [0,1,3,6,10,15,25,36,49,100]
CS = plt.contour(beta,alpha,-np.log(GCE_like_2d/GCE_like_2d.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('beta')
plt.ylabel('alpha')
plt.savefig('plots/log_parabola/'+file_name+'_GCE_contours.png')
plt.clf()



GCE_like_1d = np.trapz(np.trapz(np.exp(log_like_GCE_4d[:,:,:,0]), x=alpha, axis=1), x=beta, axis=1)

plt.plot(N0_GCE,GCE_like_1d/GCE_like_1d.max())
plt.xlabel('Normalization')
plt.xscale('log')
plt.ylabel('Scaled probability')
plt.savefig('plots/log_parabola/'+file_name+'_GCE_norm_posterior.png')
plt.clf()


evidence_GCE = np.trapz(np.trapz(GCE_like_2d,x = alpha,axis=0),x=beta,axis=0)/alpha_prior_norm/beta_prior_norm



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



N0_dwarf = np.logspace(-11,0,150)
n_N0_dwarf = len(N0_dwarf)
N0_dwarf_normalization = np.trapz(np.ones(n_N0_dwarf), x=np.log(N0_dwarf))
like_dwarf_2d = np.ones((n_alpha,n_beta))
for name in like_name:
    print name
    data_energy_dwarf = np.loadtxt('release-01-00-02/'+name+'.txt')
    emin_dwarf = np.unique(data_energy_dwarf[:,0])/1000.
    emax_dwarf = np.unique(data_energy_dwarf[:,1])/1000. #delete copies and convert from MeV to GeV
    ebin_dwarf = np.sqrt(emin_dwarf*emax_dwarf)
    ebin_dwarf = np.tile(ebin_dwarf[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0_dwarf,n_alpha,n_beta,n_eb,1))
    nflux_dwarf = GCE_calcs.calculations.get_spec_log_parab(N0_dwarf,alpha,beta,Eb,emax_dwarf,emin_dwarf)
    eflux_dwarf = nflux_dwarf*ebin_dwarf#####FIX THISSSS !!!!!!!
    log_like_dwarf_5d = GCE_calcs.analysis.dwarf_delta_log_like_log_parab(eflux_dwarf,name)
    like_dwarf_4d = np.exp(np.sum(log_like_dwarf_5d,axis=4))
    like_dwarf_2d_N0_beta = np.trapz(like_dwarf_4d[:,:,:,0],x=alpha,axis=1)
    like_dwarf_2d_N0_alpha = np.trapz(like_dwarf_4d[:,:,:,0],x=beta,axis=2)
    CS = plt.contour(beta,N0_dwarf,-np.log(like_dwarf_2d_N0_beta/like_dwarf_2d_N0_beta.max()),levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('beta')
    plt.yscale('log')
    plt.ylabel('Normalization')
    plt.savefig('plots/log_parabola/'+file_name+'_'+name+'_dwarf_N0_beta_contours.png')
    plt.clf()
    CS = plt.contour(alpha,N0_dwarf,-np.log(like_dwarf_2d_N0_alpha/like_dwarf_2d_N0_alpha.max()),levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('alpha')
    plt.ylabel('Normalization')
    plt.yscale('log')
    plt.savefig('plots/log_parabola/'+file_name+'_'+name+'_dwarf_N0_alpha_contours.png')
    plt.clf()
    like_dwarf_2d *= np.trapz(like_dwarf_4d[:,:,:,0],x=np.log(N0_dwarf),axis=0)/N0_dwarf_normalization
    dwarf_norm_posterior = np.trapz(np.trapz(like_dwarf_4d[:,:,:,0] , x=alpha, axis=1), x=beta, axis=1 )
    plt.plot(N0_dwarf,dwarf_norm_posterior/dwarf_norm_posterior.max())
    plt.xlabel('Normalization')
    plt.xscale('log')
    plt.ylabel('Scaled Probability')
    plt.ylim(0,1.1)
    plt.savefig('plots/log_parabola/'+file_name+'_'+name+'_norm_posterior.png')
    plt.clf()

CS = plt.contour(beta,alpha,-np.log(like_dwarf_2d/like_dwarf_2d.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('beta')
plt.ylabel('alpha')
plt.savefig('plots/log_parabola/'+file_name+'_dwarf_contours.png')
plt.clf()


evidence_dwarf =  np.trapz(np.trapz(like_dwarf_2d ,x = alpha,axis=0),x=beta,axis=0)/alpha_prior_norm/beta_prior_norm

print 'the GCE evidence is ' +str(evidence_GCE)
print 'the dwarf evidence is '+str(evidence_dwarf)

####################
####### C-C-C-COMBO
####################

combo_like_2d = like_dwarf_2d *GCE_like_2d


CS = plt.contour(beta,alpha,-np.log(combo_like_2d/combo_like_2d.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('beta')
plt.ylabel('alpha')
plt.savefig('plots/log_parabola/'+file_name+'_combo_contours.png')
plt.clf()


evidence_combo = np.trapz(np.trapz(combo_like_2d,x = alpha, axis = 0),x=beta,axis=0)/alpha_prior_norm/beta_prior_norm

print 'the product of the dwarf evidence and GCE evidence is ' + str(evidence_dwarf*evidence_GCE)

print 'the combined evidence is ' +str(evidence_combo)
