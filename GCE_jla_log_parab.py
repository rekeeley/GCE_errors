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

Eb = np.logspace(-1,0.5,40)
alpha = np.linspace(-3,1.5,40)
beta = np.linspace(0.01,2,40)
N0_GCE = np.logspace(-8,-5,40)

binned_spectra = GCE_calcs.calculations.get_spec_log_parab(N0_GCE,alpha,beta,Eb,emax_GCE,emin_GCE)

unbinned_spectra = GCE_calcs.calculations.get_dn_de_log_parab(N0_GCE,alpha,beta,Eb,10**bin_center)

test_spec_type = GCE_calcs.calculations.get_spec_log_parab_for_minimizing(3.e-6,-1.8,.775,0.1859,emax_GCE,emin_GCE)

print test_spec_type

#print binned_spectra[0,0,0,0,:]
#print binned_spectra[0,0,0,:,0]
#print binned_spectra[0,0,:,0,0]
#print binned_spectra[0,:,0,0,0]
#print binned_spectra[:,0,0,0,0]


n_eb = len(Eb)
n_beta = len(beta)
n_alpha = len(alpha)
n_N0 = len(N0_GCE)

background = np.tile(background[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
exposure = np.tile(exposure[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
k = np.tile(k[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))

mu_GCE = GCE_calcs.calculations.get_mu_log_parab(background,exposure,binned_spectra)

#print mu_GCE[0,0,0,:,0]
#print mu_GCE[0,0,:,0,0]
#print mu_GCE[0,:,0,0,0]
#print mu_GCE[:,0,0,0,0]

#print mu_GCE[:,0,0,0,:]

#print background[0,0,0,0,:]
#print k[0,0,0,0,:] - background[0,0,0,0,:]



log_like_GCE_5d = GCE_calcs.analysis.poisson_log_like(k,mu_GCE)

log_like_GCE_4d = np.sum(log_like_GCE_5d,axis=4)

#print log_like_GCE_4d

max_index = np.unravel_index(log_like_GCE_4d.argmax(),log_like_GCE_4d.shape)

#print 'the index of the max prob is ' + str(max_index)

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

#plt.plot(10**bin_center,background[0,0,0,0,:],label = 'background')
plt.errorbar(10**bin_center,(k[0,0,0,0,:] - background[0,0,0,0,:])/exposure[0,0,0,0,:],yerr = np.sqrt(k[0,0,0,0,:])/exposure[0,0,0,0,:],label = 'observed residual')
plt.plot(10**bin_center, binned_spectra[max_index[0],max_index[1],max_index[2],max_index[3],:],label = 'max binned DM number flux')
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1],max_index[2]+2,max_index[3],:],label = 'beta = ' +  str(beta[max_index[2]+2]))
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1],max_index[2]-2,max_index[3],:],label = 'beta = ' +  str(beta[max_index[2]-2]))
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1]+2,max_index[2],max_index[3],:],label = 'alpha = ' +  str(alpha[max_index[1]+2]))
plt.plot(10**bin_center,binned_spectra[max_index[0],max_index[1]-2,max_index[2],max_index[3],:],label = 'alpha = ' + str(alpha[max_index[1]-2]))
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Flux [cm^-2 sec^-1]')
plt.legend(loc='best')
plt.xscale('log')
#plt.yscale('log')
plt.savefig('residual_test.png')
plt.clf()


#plt.plot(10**bin_center,unbinned_spectra[max_index[0],max_index[1],max_index[2],max_index[3],:])
#plt.plot(10**bin_center,unbinned_spectra[max_index[0],max_index[1],max_index[2]+2,max_index[3],:])
#plt.plot(10**bin_center,unbinned_spectra[max_index[0],max_index[1],max_index[2]-2,max_index[3],:])
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig('unbinned_spectra.png')
#plt.clf()

#########################
### plotting slices of the 4-d log-like
#########################


levels = [0,1,3,6,10,15,25,36,49]
plt.contour(alpha,beta,-log_like_GCE_4d[max_index[0],:,:,max_index[3]].T  + log_like_GCE_4d.max(),levels)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.savefig('alpha_beta_slice.png')
plt.clf()

max_index_alpha_beta = np.unravel_index(log_like_GCE_4d[max_index[0],:,:,max_index[3]].argmax(),log_like_GCE_4d[max_index[0],:,:,max_index[3]].shape)

#print 'the index of the max prob in the alpha_beta slice is ' + str(max_index_alpha_beta)



plt.contour(alpha,Eb, -log_like_GCE_4d[max_index[0],:,max_index[2],:].T + log_like_GCE_4d.max(),levels)
plt.xlabel('alpha')
plt.ylabel('Eb')
plt.yscale('log')
plt.savefig('alpha_Eb_slice.png')
plt.clf()

max_index_alpha_Eb = np.unravel_index(log_like_GCE_4d[max_index[0],:,max_index[2],:].argmax(),log_like_GCE_4d[max_index[0],:,max_index[2],:].shape)

#print 'the index of the max prob in the alpha_Eb slice is ' + str(max_index_alpha_Eb)




plt.contour(N0_GCE,alpha, -log_like_GCE_4d[:,:,max_index[2],max_index[3]].T + log_like_GCE_4d.max(),levels)
plt.xlabel('N0')
plt.xscale('log')
plt.ylabel('alpha')
plt.savefig('N0_alpha_slice.png')
plt.clf()

max_index_N0_alpha = np.unravel_index(log_like_GCE_4d[:,:,max_index[2],max_index[3]].argmax(),log_like_GCE_4d[:,:,max_index[2],max_index[3]].shape)

#print 'the index of the max prob in the N0_alpha slice is ' + str(max_index_N0_alpha)


plt.contour(N0_GCE,beta, -log_like_GCE_4d[:,max_index[1],:,max_index[3]].T + log_like_GCE_4d.max(),levels)
plt.xlabel('N0')
plt.xscale('log')
plt.ylabel('beta')
plt.savefig('N0_beta_slice.png')
plt.clf()

max_index_N0_beta = np.unravel_index(log_like_GCE_4d[:,max_index[1],:,max_index[3]].argmax(),log_like_GCE_4d[:,max_index[1],:,max_index[3]].shape)

#print 'the index of the max prob in the N0_beta slice is ' + str(max_index_N0_beta)


plt.contour(N0_GCE,Eb, -log_like_GCE_4d[:,max_index[1],max_index[2],:].T + log_like_GCE_4d.max(),levels)
plt.xlabel('N0')
plt.xscale('log')
plt.ylabel('Eb')
plt.yscale('log')
plt.savefig('N0_Eb_slice.png')
plt.clf()

max_index_N0_Eb = np.unravel_index(log_like_GCE_4d[:,max_index[1],max_index[2],:].argmax(),log_like_GCE_4d[:,max_index[1],max_index[2],:].shape)

#print 'the index of the max prob in the N0_Eb slice is ' + str(max_index_N0_Eb)


plt.contour(beta,Eb, -log_like_GCE_4d[max_index[0],max_index[1],:,:].T + log_like_GCE_4d.max(),levels)
plt.xlabel('beta')
plt.ylabel('Eb')
plt.yscale('log')
plt.savefig('beta_Eb_slice.png')
plt.clf()

max_index_beta_Eb = np.unravel_index(log_like_GCE_4d[max_index[0],max_index[1],:,:].argmax(),log_like_GCE_4d[max_index[0],max_index[1],:,:].shape)

#print 'the index of the max prob in the beta_Eb slice is ' + str(max_index_beta_Eb)


like_GCE_4d = np.exp(log_like_GCE_4d)

zero_test = np.count_nonzero(like_GCE_4d)

#print 'the number of non-zero elements are' + str(zero_test)

N0_prior_norm = np.trapz(np.ones(n_N0),x = np.log(N0_GCE))

#print 'the norm of the N0 prior is '+ str(N0_prior_norm)

like_GCE_3d = np.trapz(like_GCE_4d, x = np.log(N0_GCE),axis=0) / N0_prior_norm

#print like_GCE_3d

Eb_prior_norm = np.trapz(np.ones(n_eb),x = np.log(Eb))

#print 'the norm of the E_b prior is '+ str(Eb_prior_norm)

like_GCE_2d = np.trapz(like_GCE_3d, x = np.log(Eb),axis = 2) / Eb_prior_norm

zero_test_2d = np.count_nonzero(like_GCE_2d)

#print 'the number of non-zero elements are' + str(zero_test_2d)

alpha_prior_norm = np.trapz(np.ones(n_alpha),x = alpha)

#print 'the norm of the alpha prior is '+ str(alpha_prior_norm)

like_GCE_1d = np.trapz(like_GCE_2d, x = alpha, axis = 0)/alpha_prior_norm

beta_prior_norm = np.trapz(np.ones(n_beta), x = beta)

#print 'the norm of the beta prior is '+ str(beta_prior_norm)

like_GCE = np.trapz(like_GCE_1d, x = beta)/beta_prior_norm

print like_GCE

#print like_GCE_2d

delta_log_like_2d = -(np.log(like_GCE_2d) - np.log(like_GCE_2d.max()))

#print delta_log_like_2d.shape


posterior_N0 = np.trapz(np.trapz(np.trapz(like_GCE_4d,x = Eb,axis=3),x=beta,axis=2),x=alpha,axis=1)/alpha_prior_norm/beta_prior_norm/Eb_prior_norm

posterior_N0_beta = np.trapz(np.trapz(like_GCE_4d,x = Eb,axis=3),x=alpha,axis=1)

plt.plot(N0_GCE,posterior_N0 / posterior_N0.max())
plt.xscale('log')
plt.xlabel('Normalization')
plt.ylabel('scaled probability')
plt.savefig('N0_posterior.png')
plt.clf()

levels = [0,1,3,6,10]
plt.contour(N0_GCE,beta,-np.log(posterior_N0_beta / posterior_N0_beta.max()))
plt.xscale('log')
plt.xlabel('normalization')
plt.ylabel('beta')
plt.savefig('N0_beta_posterior.png')
plt.clf()

#print alpha
#print beta

#print delta_log_like_2d

#print np.argmax(like_GCE_2d,axis=0)

plt.contour(alpha, beta, delta_log_like_2d ,levels)
#plt.xlim(0,2)
#plt.ylim(6,8)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.savefig('posterior_test.png')
plt.clf()



###################
###### dwarfs part
###################

#Eb = np.logspace(-2,2,200)
#alpha = np.linspace(0,10,100)
#beta = np.linspace(0,10,100)
N0_dwarfs = np.logspace(-3,4,70)


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

