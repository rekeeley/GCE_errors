import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

file_path = np.array(['full','noMG','IC'])[model] +'_'+str(trunc)+'_'+str(dataset)

#file_name = np.array(['exp_cutoff_full','exp_cutoff_noMG','exp_cutoff_IC'])[model]
               
bin_center = raw[trunc:dataset,0]#logarthmic bin center
log_bin_width = bin_center[1] - bin_center[0]
emin_GCE = 10**(bin_center - log_bin_width)
emax_GCE = 10**(bin_center + log_bin_width)
k = raw[trunc:dataset,5]
background = raw[trunc:dataset,7]
exposure = raw[trunc:dataset,6]



gamma = np.linspace(-0.99, 1.5, 100)
p = np.logspace(-1, 0.5, 100)
N0_GCE = np.logspace(-8, -3, 150)

n_gamma = len(gamma)
n_p = len(p)
n_N0 = len(N0_GCE)

gamma_prior_norm = np.trapz(np.ones(n_gamma), x = gamma)
p_prior_norm = np.trapz(np.ones(n_p), x=np.log(p))
N0_prior_norm = np.trapz(np.ones(n_N0),x = np.log(N0_GCE))

binned_spectra = GCE_calcs.calculations.get_spec_exp_cutoff(N0_GCE,gamma,p,emax_GCE,emin_GCE)
# 4-d array of shape n_N0, n_alpha, n_beta, n_Eb, n_spec.  Is the number flux

background = np.tile(background[np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_gamma,n_p,1))
exposure = np.tile(exposure[np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_gamma,n_p,1))
k = np.tile(k[np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_gamma,n_p,1))

mu_GCE = GCE_calcs.calculations.get_mu_log_parab(background,exposure,binned_spectra)

log_like_GCE_4d = GCE_calcs.analysis.poisson_log_like(k,mu_GCE)

log_like_GCE_3d = np.sum(log_like_GCE_4d,axis=3)

max_index_GCE = np.unravel_index(log_like_GCE_3d.argmax(),log_like_GCE_3d.shape)
print max_index_GCE
plt.errorbar(10**bin_center,k[0,0,0,:]-background[0,0,0,:],yerr = np.sqrt(k[0,0,0,:]),label = 'Observed Residual')
plt.plot(10**bin_center,mu_GCE[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2],:]-background[0,0,0,:],label = 'Expected Residual')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.legend(loc='best')
plt.savefig('plots/exp_cutoff/'+file_path+'/test_residuals_GCE.png')
plt.clf()

like_GCE_2d_N0_p = np.trapz(np.exp(log_like_GCE_3d), x=gamma, axis=1)
like_GCE_2d_N0_gamma = np.trapz(np.exp(log_like_GCE_3d), x=np.log(p), axis=2)


levels = [0,1,3,6,10,15]
cmap = cm.cool

CS = plt.contour(p,N0_GCE,-np.log(like_GCE_2d_N0_p/like_GCE_2d_N0_p.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel(r'$E_p$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Normalization')
plt.savefig('plots/exp_cutoff/'+file_path+'/GCE_N0_p_contours.png')
plt.clf()

CS = plt.contour(gamma,N0_GCE,-np.log(like_GCE_2d_N0_gamma/like_GCE_2d_N0_gamma.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('gamma')
plt.ylabel('Normalization')
plt.yscale('log')
plt.savefig('plots/exp_cutoff/'+file_path+'/GCE_N0_gamma_contours.png')
plt.clf()

GCE_like_2d = np.trapz(np.exp(log_like_GCE_3d), x=np.log(N0_GCE), axis=0)/N0_prior_norm

CS = plt.contour(p, gamma, -np.log(GCE_like_2d/GCE_like_2d.max()), levels, cmap=cm.get_cmap(cmap,len(levels)-1))
plt.clabel(CS, inline=1, fontsize=10)
plt.xscale('log')
plt.xlabel(r'$E_p$')
plt.ylabel(r'$\gamma$')
plt.title(r'GCE $-\Delta$Log-Likelihood Contours')
plt.savefig('plots/exp_cutoff/'+file_path+'/GCE_gamma_p_contours.png')
plt.clf()



GCE_like_1d = np.trapz(np.trapz(np.exp(log_like_GCE_3d), x=gamma, axis=1), x=np.log(p), axis=1)

plt.plot(N0_GCE,GCE_like_1d/GCE_like_1d.max())
plt.xlabel('Normalization')
plt.xscale('log')
plt.ylabel('Scaled probability')
plt.savefig('plots/exp_cutoff/'+file_path+'/GCE_norm_posterior.png')
plt.clf()


evidence_GCE = np.trapz(np.trapz(GCE_like_2d, x=gamma, axis=0), x=np.log(p), axis=0)/gamma_prior_norm/p_prior_norm



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



N0_dwarf = np.logspace(-14,-7,150)
n_N0_dwarf = len(N0_dwarf)
N0_dwarf_normalization = np.trapz(np.ones(n_N0_dwarf), x=np.log(N0_dwarf))
like_dwarf_2d = np.ones((n_gamma,n_p))
for name in like_name:
    print name
    data_energy_dwarf = np.loadtxt('release-01-00-02/'+name+'.txt')
    emin_dwarf = np.unique(data_energy_dwarf[:,0])/1000.
    emax_dwarf = np.unique(data_energy_dwarf[:,1])/1000. #delete copies and convert from MeV to GeV
    ebin_dwarf = np.sqrt(emin_dwarf*emax_dwarf)
    ebin_dwarf = np.tile(ebin_dwarf[np.newaxis,np.newaxis,np.newaxis,:],(n_N0_dwarf, n_gamma, n_p, 1))
    nflux_dwarf = GCE_calcs.calculations.get_spec_exp_cutoff(N0_dwarf, gamma, p, emax_dwarf, emin_dwarf)
    eflux_dwarf = nflux_dwarf*ebin_dwarf#####FIX THISSSS !!!!!!!
    log_like_dwarf_4d = GCE_calcs.analysis.dwarf_delta_log_like(eflux_dwarf,name)
    like_dwarf_3d = np.exp(np.sum(log_like_dwarf_4d, axis=3))
    like_dwarf_2d_N0_p = np.trapz(like_dwarf_3d, x=gamma, axis=1)
    like_dwarf_2d_N0_gamma = np.trapz(like_dwarf_3d, x=np.log(p), axis=2)
    CS = plt.contour(p, N0_dwarf, -np.log(like_dwarf_2d_N0_p/like_dwarf_2d_N0_p.max()), levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(r'$E_p$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Normalization')
    plt.savefig('plots/exp_cutoff/'+file_path+'/'+name+'_N0_p_contours.png')
    plt.clf()
    CS = plt.contour(gamma,N0_dwarf,-np.log(like_dwarf_2d_N0_gamma/like_dwarf_2d_N0_gamma.max()),levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Normalization')
    plt.yscale('log')
    plt.savefig('plots/exp_cutoff/'+file_path+'/'+name+'_N0_gamma_contours.png')
    plt.clf()
    like_dwarf_2d *= np.trapz(like_dwarf_3d, x=np.log(N0_dwarf), axis=0)/N0_dwarf_normalization
    dwarf_norm_posterior = np.trapz(np.trapz(like_dwarf_3d, x=gamma, axis=1), x=np.log(p), axis=1)
    plt.plot(N0_dwarf,dwarf_norm_posterior/dwarf_norm_posterior.max())
    plt.xlabel('Normalization')
    plt.xscale('log')
    plt.ylabel('Scaled Probability')
    plt.ylim(0,1.1)
    plt.savefig('plots/exp_cutoff/'+file_path+'/'+name+'_norm_posterior.png')
    plt.clf()

CS = plt.contour(p, gamma, -np.log(like_dwarf_2d/like_dwarf_2d.max()), levels, cmap=cm.get_cmap(cmap,len(levels)-1))
plt.clabel(CS, inline=1, fontsize=10)
plt.xscale('log')
plt.xlabel(r'$E_p$')
plt.ylabel(r'$\gamma$')
plt.title(r'Combined Dwarf $-\Delta$Log-Likelihood Contours')
plt.savefig('plots/exp_cutoff/'+file_path+'/dwarf_gamma_p_contours.png')
plt.clf()


evidence_dwarf =  np.trapz(np.trapz(like_dwarf_2d, x=gamma, axis=0), x=np.log(p), axis=0)/gamma_prior_norm/p_prior_norm



####################
####### C-C-C-COMBO
####################

combo_like_2d = like_dwarf_2d*GCE_like_2d


CS = plt.contour(p, gamma, -np.log(combo_like_2d/combo_like_2d.max()), levels, cmap=cm.get_cmap(cmap,len(levels)-1))
plt.clabel(CS, inline=1, fontsize=10)
plt.xscale('log')
plt.xlabel(r'$E_p$')
plt.ylabel(r'$\gamma$')
plt.title(r'Combined $-\Delta$Log-Likelihood Contours')
plt.savefig('plots/exp_cutoff/'+file_path+'/combo_gamma_p_contours.png')
plt.clf()


evidence_combo = np.trapz(np.trapz(combo_like_2d, x=gamma, axis=0), x=np.log(p), axis=0)/gamma_prior_norm/p_prior_norm

print 'the GCE evidence is ' +str(evidence_GCE)
print 'the dwarf evidence is '+str(evidence_dwarf)

print 'the product of the dwarf evidence and GCE evidence is ' + str(evidence_dwarf*evidence_GCE)

print 'the combined evidence is ' +str(evidence_combo)
