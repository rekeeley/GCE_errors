import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
from scipy import integrate
from matplotlib import gridspec


rc('font',**{'family':'serif','serif':['Times New Roman']})
plt.rcParams.update({'font.size': 24})

import GCE_calcs

############################
# MSP Models
############################

###################
######  GCE part
###################

##### DIFFERENT MODELS TO EXPLAIN GCE #####
channel = 0 #log-parabola
#channel = 1 #exponential cutoff

##### DIFFERENT MODELS FOR THE BACKGOUND #####
model = 0  #kwa
#model = 1 #glliem
#model = 2  #gal2yr

channel_name=['lp','ec'][channel]

model_name = ['kwa','glliem','gal2yr'][model]

raw= np.loadtxt('data/background/GC_'+model_name+'.txt')
raw=raw.T

emin_GCE = raw[:,0]/1000.
emax_GCE = raw[:,1]/1000.
bin_center = bin_center = np.sqrt(emin_GCE*emax_GCE)

k = raw[:,4]
background = raw[:,3]
exposure = raw[:,2]

if channel_name == 'lp':
    Eb = 3.
    alpha = np.linspace(2.0, 3.0, 11, endpoint=True)
    beta = np.linspace(0.4, 0.7, 13, endpoint=True)
else:
    alpha = np.linspace(-1.3, -0.3, 11, endpoint=True) #gamma
    beta = np.linspace(1., 3., 11, endpoint=True) #Ec

#now instead the total normalization of the spectra is the product of the stellar mass of the region times the the gamma-ray flux per stellar mass.  The first changes between the regions and the second is the same.  Thsi is the zeroth order ansatz
gfpsm = np.logspace(29.5, 32.5, 31, endpoint=True)

GC_SM_mu = 2.6e8
GC_SM_sigma = 0.1*GC_SM_mu
GC_dist_mu = 8.25*3.086e21
GC_dist_sigma = 0.05*GC_dist_mu

norm_GC = np.logspace(np.log10(GC_SM_mu /GC_dist_mu**2./4./np.pi)-1.5,np.log10(GC_SM_mu /GC_dist_mu**2./4./np.pi)+1.5,31,endpoint=True)

n_beta = len(beta)
n_alpha = len(alpha)
n_gfpsm = len(gfpsm)
n_SM = len(norm_GC)

beta_prior_norm = np.trapz(np.ones(n_beta), x=beta)
alpha_prior_norm = np.trapz(np.ones(n_alpha),x=alpha)
gfpsm_prior_norm = np.trapz(np.ones(n_gfpsm), x=np.log(gfpsm))


if channel_name =='lp':
    nflux_GCE = GCE_calcs.calculations.get_spec_log_parab(gfpsm,norm_GC,alpha,beta,Eb,emax_GCE,emin_GCE)
else:
    nflux_GCE = GCE_calcs.calculations.get_spec_exp_cutoff(gfpsm,norm_GC,alpha,beta,emax_GCE,emin_GCE)
# 5-d array of shape n_gfpsm, n_SM, n_alpha, n_beta, n_spec.  This is the number flux

background = np.tile(background[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_gfpsm,n_SM,n_alpha,n_beta,1))
exposure = np.tile(exposure[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_gfpsm,n_SM,n_alpha,n_beta,1))
k = np.tile(k[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_gfpsm,n_SM,n_alpha,n_beta,1))

mu_GCE = GCE_calcs.calculations.get_mu_log_parab(background,exposure,nflux_GCE)

log_like_GCE_5d = GCE_calcs.analysis.poisson_log_like(k,mu_GCE)

log_like_GCE_4d = np.sum(log_like_GCE_5d,axis=4)

GC_SM_prior = GCE_calcs.analysis.get_msp_prior(norm_GC, GC_SM_mu, GC_SM_sigma, GC_dist_mu, GC_dist_sigma)
norm_test = np.trapz(GC_SM_prior, x=np.log10(norm_GC))
assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the SM is off by more than 1%'
GC_SM_prior = np.tile(GC_SM_prior[np.newaxis,:,np.newaxis,np.newaxis],(n_gfpsm,1,n_alpha,n_beta))

like_GCE_4d = np.exp(log_like_GCE_4d)*GC_SM_prior

max_index_GCE = np.unravel_index(like_GCE_4d.argmax(),like_GCE_4d.shape)

like_GCE_3d = np.trapz(like_GCE_4d, x=np.log10(norm_GC), axis=1)

like_GCE_2d_gfpsm_beta = np.trapz(like_GCE_3d, x=alpha, axis=1)
like_GCE_2d_gfpsm_alpha = np.trapz(like_GCE_3d, x=beta, axis=2)
like_GCE_2d_alpha_beta = np.trapz(like_GCE_3d, x=np.log(gfpsm), axis=0)

like_GCE_1d = np.trapz(np.trapz(like_GCE_3d, x=alpha, axis=1), x=beta, axis=1)

evidence_GCE = np.trapz(np.trapz(np.trapz(like_GCE_3d, x=np.log(gfpsm), axis=0), x=alpha, axis=0), x=beta, axis=0)/beta_prior_norm/alpha_prior_norm/gfpsm_prior_norm


#######################
## Plots for GCE part
#######################

plt.plot(norm_GC,GC_SM_prior[0,:,0,0]/GC_SM_prior[0,:,0,0].max() )
plt.xscale('log')
plt.ylim(0,1.1)
plt.xlim(norm_GC[0],norm_GC[-1])
plt.xlabel(r'Norm [$M_\odot$ cm$^{-2}$]')
plt.ylabel('Scaled Probability')
plt.savefig('plots/'+channel_name+'/'+model_name+'/norm.png')
plt.clf()

plt.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.96)
plt.errorbar(bin_center, k[0,0,0,0,:]-background[0,0,0,0,:], yerr=np.sqrt(k[0,0,0,0,:]), color='c', label='Observed Residual',linewidth=2.0)
plt.plot(bin_center, mu_GCE[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2],max_index_GCE[3],:]-background[0,0,0,0,:], color='m', label='Expected Residual',linewidth=2.0)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.2,1e2)
plt.ylim(1e1,1e5)
plt.xlabel('Energy [GeV]')
plt.ylabel('Number counts')
plt.legend(loc='upper right',frameon=False,fontsize=22)
plt.savefig('plots/'+channel_name+'/'+model_name+'/residuals_GCE.pdf')
plt.clf()

np.savetxt('plots/'+channel_name+'/'+model_name+'/residuals_GCE.txt', (bin_center, k[0,0,0,0,:], background[0,0,0,0,:], mu_GCE[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2],max_index_GCE[3],:]))

cmap = cm.cool
levels = [0,1,3,6,10,15]

CS = plt.contour(beta,gfpsm,-np.log(like_GCE_2d_gfpsm_beta/like_GCE_2d_gfpsm_beta.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('beta')
plt.yscale('log')
plt.ylabel(r'Gamma ray rate per stellar mass [s$^{-1}$ M$_\odot$]')
plt.savefig('plots/'+channel_name+'/'+model_name+'/GCE_gfpsm_beta_contours.png')
plt.clf()

CS = plt.contour(alpha,gfpsm,-np.log(like_GCE_2d_gfpsm_alpha/like_GCE_2d_gfpsm_alpha.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'Gamma ray rate per stellar mass [s$^{-1}$ M$_\odot$]')
plt.yscale('log')
plt.savefig('plots/'+channel_name+'/'+model_name+'/GCE_gfpsm_alpha_contours.png')
plt.clf()

CS = plt.contour(beta, alpha, -np.log(like_GCE_2d_alpha_beta/like_GCE_2d_alpha_beta.max()), levels, cmap=cm.get_cmap(cmap,len(levels)-1))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\alpha$')
plt.title(r'GCE -$\Delta$Log-Likelihood Contours')
plt.savefig('plots/'+channel_name+'/'+model_name+'/GCE_alpha_beta_contours.png')
plt.clf()

plt.subplots_adjust(left=0.12, bottom=0.14, right=0.96, top=0.96)
plt.plot(gfpsm,like_GCE_1d/like_GCE_1d.max(),label='GCE', linewidth=2.0)
plt.xlabel(r'Gamma ray rate per stellar mass [s$^{-1}$ M${_\odot}{^{-1}}$]')
plt.xscale('log')
plt.ylim(0,1.1)
plt.ylabel('Scaled probability')
plt.legend(loc='upper right',frameon=False,fontsize=22)
plt.savefig('plots/'+channel_name+'/'+model_name+'/GCE_norm_posterior.pdf')
plt.clf()





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


#arxiv.org/pdf/1204.1562.pdf

dwarf_SM_mu = 1.e6*np.array([ 0.029,
                            0.23,
                            0.0079,
                            0.38,
                            0.0037,
                            0.29,
                            20,
                            0.037,
                            5.5,
                            0.74,
                            0.019,
                            2.3,
                            0.00034,
                            0.44,
                            0.014,
                            0.0041,
                            0.29,
                            0.0010])

dwarf_dist_mu = 3.086e21*np.array([ 66.,
                            218.,
                            160.,
                            105.,
                            44.,
                            76.,
                            147.,
                            132.,
                            254.,
                            233.,
                            154.,
                            86.,
                            23.,
                            86.,
                            97.,
                            32.,
                            76.,
                            38.])# convert kpc to cm

dwarf_dist_sigma = 3.086e21*np.array([2.,
                            10.,
                            4.,
                            6.,
                            4.,
                            6.,
                            12.,
                            12.,
                            15.,
                            14.,
                            6.,
                            6.,
                            2.,
                            4.,
                            4.,
                            4.,
                            3.,
                            7.])


like_dwarf_3d = np.ones((n_gfpsm,n_alpha,n_beta))
for i in range(len(like_name)):
    name = like_name[i]
    print name
    norm_dwarf = np.logspace(np.log10(dwarf_SM_mu[i] /dwarf_dist_mu[i]**2./4/np.pi)-1.5,np.log10(dwarf_SM_mu[i] /dwarf_dist_mu[i]**2./4/np.pi)+1.5,31,endpoint=True)
    n_norm_dwarf = len(norm_dwarf)
    data_energy_dwarf = np.loadtxt('release-01-00-02/'+name+'.txt')
    emin_dwarf = np.unique(data_energy_dwarf[:,0])/1000.
    emax_dwarf = np.unique(data_energy_dwarf[:,1])/1000. #delete copies and convert from MeV to GeV
    ebin_dwarf = np.sqrt(emin_dwarf*emax_dwarf)
    ebin_dwarf = np.tile(ebin_dwarf[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_gfpsm,n_norm_dwarf,n_alpha,n_beta,1))
    if channel_name =='lp':
        nflux_dwarf = GCE_calcs.calculations.get_spec_log_parab(gfpsm,norm_dwarf,alpha,beta,Eb,emax_dwarf,emin_dwarf)
    else:
        nflux_dwarf = GCE_calcs.calculations.get_spec_exp_cutoff(gfpsm,norm_dwarf,alpha,beta,emax_dwarf,emin_dwarf)
    eflux_dwarf = nflux_dwarf*ebin_dwarf#####FIX THISSSS !!!!!!!
    log_like_dwarf_5d = GCE_calcs.analysis.dwarf_delta_log_like_log_parab(eflux_dwarf,name)
    log_like_dwarf_4d = np.sum(log_like_dwarf_5d,axis=4)
    norm_dwarf_prior = GCE_calcs.analysis.get_msp_prior(norm_dwarf, dwarf_SM_mu[i], 0.1*dwarf_SM_mu[i], dwarf_dist_mu[i], dwarf_dist_sigma[i])
    plt.plot(norm_dwarf,norm_dwarf_prior/norm_dwarf_prior.max() )
    plt.xscale('log')
    plt.ylim(0,1.1)
    plt.xlim(norm_dwarf[0],norm_dwarf[-1])
    plt.xlabel('Norm')
    plt.ylabel('Scaled Probability')
    plt.savefig('plots/'+channel_name+'/'+model_name+'/'+name+'_norm.png')
    plt.clf()
    norm_test = np.trapz(norm_dwarf_prior, x=np.log10(norm_dwarf))
    assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the SM is off by more than 1%'
    norm_dwarf_prior = np.tile(norm_dwarf_prior[np.newaxis,:,np.newaxis,np.newaxis],(n_gfpsm,1,n_alpha,n_beta))
    like_dwarf_4d = np.exp(log_like_dwarf_4d)*norm_dwarf_prior
    like_dwarf_3d_temp = np.trapz(like_dwarf_4d, x=np.log10(norm_dwarf), axis=1)
    like_dwarf_2d_gfpsm_beta = np.trapz(like_dwarf_3d_temp, x=alpha, axis=1)
    like_dwarf_2d_gfpsm_alpha = np.trapz(like_dwarf_3d_temp, x=beta, axis=2)
    like_dwarf_1d_gfpsm = np.trapz(np.trapz(like_dwarf_3d_temp, x=alpha, axis=1), x=beta, axis=1)
    CS = plt.contour(beta,gfpsm,-np.log(like_dwarf_2d_gfpsm_beta/like_dwarf_2d_gfpsm_beta.max()),levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(r'$\beta$')
    plt.yscale('log')
    plt.ylabel(r'Gamma ray rate per stellar mass [s$^{-1}$ M$_\odot$]')
    plt.savefig('plots/'+channel_name+'/'+model_name+'/'+name+'_dwarf_gfpsm_beta_contours.png')
    plt.clf()
    CS = plt.contour(alpha,gfpsm,-np.log(like_dwarf_2d_gfpsm_alpha/like_dwarf_2d_gfpsm_alpha.max()),levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'Gamma ray rate per stellar mass [s$^{-1}$ M$_\odot$]')
    plt.yscale('log')
    plt.savefig('plots/'+channel_name+'/'+model_name+'/'+name+'_dwarf_gfpsm_alpha_contours.png')
    plt.clf()
    plt.plot(gfpsm,like_dwarf_1d_gfpsm/like_dwarf_1d_gfpsm.max())
    plt.xlabel(r'Gamma ray rate per stellar mass [s$^{-1}$ M$_\odot$]')
    plt.xscale('log')
    plt.ylabel('Scaled Probability')
    plt.ylim(0,1.1)
    plt.savefig('plots/'+channel_name+'/'+model_name+'/'+name+'_gfpsm_posterior.png')
    plt.clf()
    like_dwarf_3d *= like_dwarf_3d_temp


like_dwarf_2d = np.trapz(like_dwarf_3d, x=np.log(gfpsm), axis=0)

CS = plt.contour(beta,alpha,-np.log(like_dwarf_2d/like_dwarf_2d.max()),levels, cmap=cm.get_cmap(cmap,len(levels)-1))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\alpha$')
plt.title(r'Combined Dwarf $-\Delta$Log-Likelihood Contours')
plt.savefig('plots/'+channel_name+'/'+model_name+'/dwarf_alpha_beta_contours.png')
plt.clf()


like_dwarf_1d_gfpsm = np.trapz(np.trapz(like_dwarf_3d, x=alpha, axis=1), x=beta, axis=1)

plt.subplots_adjust(left=0.12, bottom=0.14, right=0.96, top=0.96)
plt.plot(gfpsm,like_dwarf_1d_gfpsm/like_dwarf_1d_gfpsm.max(), 'c', label='Combined dwarfs', linewidth=2.0)
plt.plot(gfpsm,like_GCE_1d/like_GCE_1d.max(), 'm', label='GCE', linewidth=2.0)
plt.xlabel(r'Gamma ray rate per stellar mass [s$^{-1}$ M${_\odot}{^{-1}}$]')
plt.xscale('log')
plt.ylabel('Scaled Probability')
plt.ylim(0,1.1)
plt.legend(loc = 'upper right', frameon=False, fontsize=22)
plt.savefig('plots/'+channel_name+'/'+model_name+'/gfpsm_posteriors.pdf')
plt.clf()


evidence_dwarf =  np.trapz(np.trapz(np.trapz(like_dwarf_3d, x=np.log(gfpsm), axis=0), x=alpha, axis=0), x=beta, axis=0)/alpha_prior_norm/beta_prior_norm/gfpsm_prior_norm



####################
####### C-C-C-COMBO
####################

combo_like_3d = like_dwarf_3d * like_GCE_3d

combo_like_2d = np.trapz(combo_like_3d, x=np.log(gfpsm), axis=0)

CS = plt.contour(beta,alpha,-np.log(combo_like_2d/combo_like_2d.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('beta')
plt.ylabel('alpha')
plt.savefig('plots/'+channel_name+'/'+model_name+'/combo_alpha_beta_contours.png')
plt.clf()


evidence_combo = np.trapz(np.trapz(np.trapz(combo_like_3d, x=np.log(gfpsm), axis=0), x=alpha, axis=0), x=beta, axis=0)/alpha_prior_norm/beta_prior_norm/gfpsm_prior_norm

print 'the GCE evidence is ' +str(evidence_GCE)
print 'the dwarf evidence is '+str(evidence_dwarf)

print 'the product of the dwarf evidence and GCE evidence is ' + str(evidence_dwarf*evidence_GCE)

print 'the combined evidence is ' +str(evidence_combo)

np.savetxt('plots/'+channel_name+'/'+model_name+'/log_parab_evidences.txt',np.array([evidence_GCE, evidence_dwarf, evidence_dwarf*evidence_GCE, evidence_combo, evidence_dwarf*evidence_GCE/evidence_combo, like_GCE_4d.max()]) )
