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

binned_spectra = GCE_calcs.calculations.get_spec_log_parab(N0_GCE,alpha,beta,Eb,emax_GCE,emin_GCE)

unbinned_spectra = GCE_calcs.calculations.get_dn_de_log_parab(N0_GCE,alpha,beta,Eb,10**bin_center)

test_spec_type = GCE_calcs.calculations.get_spec_log_parab_for_minimizing(3.e-6,-1.8,.775,0.1859,emax_GCE,emin_GCE)


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


max_index = np.unravel_index(log_like_GCE_4d.argmax(),log_like_GCE_4d.shape)

print 'the index of the max prob is ' + str(max_index)

print 'the max alpha is ' + str(alpha[max_index[1]])
print 'the max beta is ' + str(beta[max_index[2]])
print 'the max Eb is ' + str(Eb[max_index[3]])
print 'the max N0 is ' + str(N0_GCE[max_index[0]])


plt.plot(10**bin_center,background[0,0,0,0,:],label = 'background')
plt.errorbar(10**bin_center,k[0,0,0,0,:],yerr = np.sqrt(k[0,0,0,0,:]),label = 'observed counts')
plt.plot(10**bin_center, mu_GCE[max_index[0],max_index[1],max_index[2],max_index[3],:],label = 'expected counts')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')
plt.savefig('log_parab_test.png')
plt.clf()


plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1],max_index[2],max_index[3],:],label = 'maximum')
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1],max_index[2]+2,max_index[3],:],label = 'beta = ' + str(beta[max_index[2]+2]))
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1],max_index[2]-2,max_index[3],:],label = 'beta = ' + str(beta[max_index[2]-2]))
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1]+2,max_index[2],max_index[3],:],label = 'alpha =  ' + str(alpha[max_index[2]+2]))
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1]-2,max_index[2],max_index[3],:],label = 'alpha = ' + str(alpha[max_index[2]-2]))
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')
plt.savefig('beta_test.png')
plt.clf()

plt.errorbar(10**bin_center,(k[0,0,0,0,:] - background[0,0,0,0,:])/exposure[0,0,0,0,:],yerr = np.sqrt(k[0,0,0,0,:])/exposure[0,0,0,0,:],label = 'observed residual')
plt.plot(10**bin_center, binned_spectra[max_index[0],max_index[1],max_index[2],max_index[3],:],label = 'max binned DM number flux')
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1],max_index[2]+2,max_index[3],:],label = 'beta = ' +  str(beta[max_index[2]+2]))
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1],max_index[2]-2,max_index[3],:],label = 'beta = ' +  str(beta[max_index[2]-2]))
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1]+2,max_index[2],max_index[3],:],label = 'alpha = ' +  str(alpha[max_index[1]+2]))
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1]-2,max_index[2],max_index[3],:],label = 'alpha = ' + str(alpha[max_index[1]-2]))
#plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1],max_index[2],max_index[3]+2,:],label = 'Eb = ' + str(Eb[max_index[3]+2]))
#plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1],max_index[2],max_index[3]-2,:],label = 'Eb = ' + str(Eb[max_index[3]-2]))
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Flux [cm^-2 sec^-1]')
plt.legend(loc='best')
plt.xscale('log')
plt.savefig('residual_test.png')
plt.clf()



like_GCE_4d = np.exp(log_like_GCE_4d)

N0_prior_norm = np.trapz(np.ones(n_N0),x = np.log(N0_GCE))

like_GCE_3d = np.trapz(like_GCE_4d, x = np.log(N0_GCE),axis=0) / N0_prior_norm

like_GCE_2d = like_GCE_3d[:,:,0]

alpha_prior_norm = np.trapz(np.ones(n_alpha),x = alpha)

like_GCE_1d = np.trapz(like_GCE_2d, x = alpha, axis = 0)/alpha_prior_norm

beta_prior_norm = np.trapz(np.ones(n_beta), x = beta)

like_GCE = np.trapz(like_GCE_1d, x = beta)/beta_prior_norm

print like_GCE

posterior_N0 = np.trapz(np.trapz(like_GCE_4d[:,:,:,0],x = beta, axis = 2),x = alpha, axis = 1)

posterior_alpha = np.trapz(np.trapz(like_GCE_4d[:,:,:,0],x = np.log(N0_GCE),axis=0),x = beta, axis=1)

posterior_beta = np.trapz(np.trapz(like_GCE_4d[:,:,:,0],x = np.log(N0_GCE),axis=0),x = alpha, axis=0)

posterior_N0_beta = np.trapz(like_GCE_4d[:,:,:,0],x=alpha,axis=1)

posterior_N0_alpha = np.trapz(like_GCE_4d[:,:,:,0],x = beta, axis=2)

plt.plot(N0_GCE,posterior_N0 / posterior_N0.max())
plt.xscale('log')
plt.xlabel('Normalization')
plt.ylabel('scaled probability')
plt.savefig('N0_posterior.png')
plt.clf()

plt.plot(alpha,posterior_alpha / posterior_alpha.max())
#plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('scaled probability')
plt.savefig('alpha_posterior.png')
plt.clf()

plt.plot(beta,posterior_beta / posterior_beta.max())
#plt.xscale('log')
plt.xlabel('beta')
plt.ylabel('scaled probability')
plt.savefig('beta_posterior.png')
plt.clf()


levels = [0,1,3,6,10,15,25,36,49,100]

plt.contour(N0_GCE,alpha,-np.log(posterior_N0_alpha.T  / posterior_N0_alpha.max()),levels)
plt.xscale('log')
plt.xlabel('normalization')
plt.ylabel('alpha')
plt.savefig('N0_alpha_posterior.png')
plt.clf()


#levels = [0,1,3,6,10]
plt.contour(N0_GCE,beta,-np.log(posterior_N0_beta.T  / posterior_N0_beta.max()),levels)
plt.xscale('log')
plt.xlabel('normalization')
plt.ylabel('beta')
plt.savefig('N0_beta_posterior.png')
plt.clf()

#print alpha
#print beta

#print delta_log_like_2d

#print np.argmax(like_GCE_2d,axis=0)

plt.contour(alpha, beta, -np.log(like_GCE_2d.T) + np.log(like_GCE_2d.max()) ,levels)
#plt.xlim(0,2)
#plt.ylim(6,8)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.savefig('alpha_beta_posterior.png')
plt.clf()



###################
###### dwarfs part
###################





#Eb = np.logspace(-2,2,200)
#alpha = np.linspace(0,10,100)
#beta = np.linspace(0,10,100)
data_energy_draco = np.loadtxt('release-01-00-02/like_draco.txt')
emin_dwarf = np.unique(data_energy_draco[:,0])/1000.
emax_dwarf = np.unique(data_energy_draco[:,1])/1000. #delete copies and convert from MeV to GeV
data_draco = np.loadtxt('dwarf_re_data/like_draco_data.txt')
k_draco = data_draco[:,0]
back_flux_draco = data_draco[:,1]
expo_draco = data_draco[:,2]

ecenter_dwarf = np.exp(0.5*(np.log(emax_dwarf)+ np.log(emin_dwarf)))

N0_dwarfs = np.logspace(-7,-6,100)

binned_spectra_draco = GCE_calcs.calculations.get_spec_log_parab(N0_dwarfs,alpha,beta,Eb,emax_dwarf,emin_dwarf)
mu_draco = GCE_calcs.calculations.get_mu_log_parab(expo_draco*back_flux_draco,expo_draco,binned_spectra_draco)
log_like_draco_5d = GCE_calcs.analysis.poisson_log_like(k_draco,mu_draco)

log_like_draco_4d = np.sum(log_like_draco_5d,axis=4)

max_index_draco = np.unravel_index(log_like_draco_4d.argmax(),log_like_draco_4d.shape)

like_draco_4d = np.exp(log_like_draco_4d)

plt.plot(ecenter_dwarf, expo_draco*back_flux_draco,label = 'background')
plt.plot(ecenter_dwarf, k_draco,label = 'observed number count')
plt.plot(ecenter_dwarf, mu_draco[max_index_draco[0],max_index_draco[1],max_index_draco[2],max_index_draco[3],:],label = 'predicted number count')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig('dwarf_number_counts.png')
plt.clf()

plt.plot(ecenter_dwarf, k_draco/expo_draco - back_flux_draco,label = 'observed residual')
plt.plot(ecenter_dwarf, binned_spectra_draco[max_index_draco[0],max_index_draco[1],max_index_draco[2],max_index_draco[3],:],label ='predicted residual')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Flux [cm^-2 sec^-1]')
plt.xscale('log')
plt.legend(loc='best')
plt.savefig('dwarf_residual.png')
plt.clf()

N0_draco_normalization = np.trapz(np.ones(len(N0_dwarfs)),x = np.log(N0_dwarfs))

N0_draco_posterior = np.trapz(np.trapz(like_draco_4d[:,:,:,0],x = alpha,axis=1),x = beta,axis=1)

alpha_draco_posterior = np.trapz(np.trapz(like_draco_4d[:,:,:,0],x = np.log(N0_dwarfs),axis=0),x = beta,axis=1)

beta_draco_posterior = np.trapz(np.trapz(like_draco_4d[:,:,:,0],x = np.log(N0_dwarfs),axis=0),x = alpha,axis=0)

alpha_beta_posterior = np.trapz(like_draco_4d[:,:,:,0],x = np.log(N0_dwarfs),axis=0)

N0_alpha_posterior = np.trapz(like_draco_4d[:,:,:,0],x = beta,axis=2)

N0_beta_posterior = np.trapz(like_draco_4d[:,:,:,0],x = alpha,axis=1)

dwarf_like1 = np.trapz(alpha_draco_posterior,x = alpha)/alpha_prior_norm/beta_prior_norm/N0_draco_normalization

print dwarf_like1


plt.plot(N0_dwarfs, N0_draco_posterior/N0_draco_posterior.max())
plt.xlabel('Normalization')
plt.xscale('log')
plt.ylabel('scaled probability')
plt.savefig('dwarf_N0_posterior.png')
plt.clf()

plt.plot(alpha, alpha_draco_posterior/alpha_draco_posterior.max())
plt.xlabel('alpha')
plt.ylabel('scaled probability')
plt.savefig('dwarf_alpha_posterior.png')
plt.clf()

plt.plot(beta, beta_draco_posterior/beta_draco_posterior.max())
plt.xlabel('beta')
plt.ylabel('scaled probability')
plt.savefig('dwarf_beta_posterior.png')
plt.clf()

plt.contour(N0_dwarfs,alpha,-np.log(N0_alpha_posterior.T  / N0_alpha_posterior.max()),levels)
plt.xscale('log')
plt.xlabel('normalization')
plt.ylabel('alpha')
plt.savefig('dwarf_N0_alpha_posterior.png')
plt.clf()

plt.contour(N0_dwarfs,beta,-np.log(N0_beta_posterior.T  / N0_beta_posterior.max()),levels)
plt.xscale('log')
plt.xlabel('normalization')
plt.ylabel('beta')
plt.savefig('dwarf_N0_beta_posterior.png')
plt.clf()

plt.contour(alpha,beta,-np.log(alpha_beta_posterior.T  / alpha_beta_posterior.max()),levels)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.savefig('dwarf_alpha_beta_posterior.png')
plt.clf()




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

N0_dwarf = np.logspace(-8,-5,100)
N0_dwarf_normalization = np.trapz(np.ones(len(N0_dwarf)),x = np.log(N0_dwarf))
log_factor = 0
log_like_dwarf_2d = np.zeros((n_alpha,n_beta))
for name in like_name:
    data_energy_dwarf = np.loadtxt('release-01-00-02/'+name+'txt')
    emin_dwarf = np.unique(data_energy_dwarf[:,0])/1000.
    emax_dwarf = np.unique(data_energy_dwarf[:,1])/1000. #delete copies and convert from MeV to GeV
    data_dwarf = np.loadtxt('dwarf_re_data/'+name+'.txt')
    k_dwarf = data_dwarf[:,0]
    back_flux_dwarf = data_dwarf[:,1]
    expo_dwarf = data_dwarf[:,2]
    binned_spectra_dwarf = GCE_calcs.calculations.get_spec_log_parab(N0_dwarf,alpha,beta,Eb,emax_dwarf,emin_dwarf)
    mu_dwarf = GCE_calcs.calculations.get_mu_log_parab(expo_dwarf*back_flux_dwarf,expo_dwarf,binned_spectra_dwarf)
    log_like_dwarf_5d = GCE_calcs.analysis.poisson_log_like(k_dwarf,mu_dwarf)
    log_like_dwarf_4d = np.sum(log_like_dwarf_5d,axis=4)
    print log_like_dwarf_4d.max()
    log_factor += log_like_dwarf_4d.max()
    log_like_dwarf_2d += np.log(np.trapz(np.exp(log_like_draco_4d[:,:,:,0]) - log_like_dwarf_4d.max(),x=np.log(N0_dwarf),axis=0)/N0_dwarf_normalization)

like_dwarf =  np.trapz(np.trapz(np.exp(log_like_dwarf_2d - log_like_dwarf_2d.max()),x = alpha,axis=0),x=beta,axis=0)

print log_factor
print like_dwarf

####################
####### C-C-C-COMBO
####################

combo_like_2d = like_GCE_2d*alpha_beta_posterior / N0_draco_normalization

plt.contour(alpha,beta,-np.log(combo_like_2d.T  / combo_like_2d.max()),levels)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.savefig('combo_alpha_beta_posterior.png')
plt.clf()

combo_like = np.trapz(np.trapz(combo_like_2d,x = alpha, axis = 0),x=beta,axis=0) / alpha_prior_norm / beta_prior_norm



print combo_like

