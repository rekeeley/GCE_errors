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

trunc = 8  #how many data points truncated from the front
dataset=20 #largest data point

mass_table = np.array([np.loadtxt('spectra/tau/LSP-energies-original.dat')[:,1],
                       np.loadtxt('spectra/bbar/LSP-energies.dat')[:,1]])[channel] #table of masses

file_path = np.array(['spectra/tau/output-gammayield-','spectra/bbar/muomega-gammayield-'])[channel]

output_file = np.array([['tau_full','tau_noMG','tau_IC'],['bbar_full','bbar_noMG','bbar_IC']])[channel][model]


N1_trunc=10
#N1 = len(mass_table)#number of points used in the mass axis
N1=20

N2 = 20 #number of points used in the cross-section axis
csmin = -27.
csrange = 4.  #fit the order of magnitude of the cross-section so 1e-26 to 1e-23

raw= np.array([np.loadtxt('data/background/GCE_paper_fullmodel_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_noMG_spectrum.dat'),
               np.loadtxt('data/background/GCE_paper_IC_spectrum.dat')])[model]


bin_center = raw[trunc:dataset,0]

k =raw[trunc:dataset,5]

background = raw[trunc:dataset,7]

exposure = raw[trunc:dataset,6]

binned_spectra = np.loadtxt('spectra/binned/binned_spectra_'+output_file+'_'+str(dataset)+'_'+str(trunc)+'.dat')

n_J=400
#J = np.array([1.e21,1.e22,1.e23])
J = np.logspace(20.,24.,num=n_J)
#J = np.array([1.5e22])

n_mass = len(mass_table)

n_sigma = 40
sigma = np.logspace(-27.,-24.,num=n_sigma)
#log_sigma = np.array([np.log10(3.e-26)])
#log_sigma = np.array([-23.,-24.,-25.,-26.,-27.])

#mass = np.array([10,20,30,40,50,60,70])

mu = GCE_calcs.calculations.get_mu(background,exposure, binned_spectra, J, sigma, mass_table)


plt.plot(bin_center,k,label = 'data')
plt.plot(bin_center,background,label = 'background')
for i in range(5):
    plt.plot(bin_center,mu[n_sigma/2,n_J/2,5+10*i,:],label = str(mass_table[5+10*i]))
plt.legend(loc='best')
plt.savefig('spectra_test.png')
plt.clf()

k = np.tile(k,(n_sigma,n_J,n_mass,1))

log_like = GCE_calcs.analysis.poisson_log_like(k,mu)

print log_like.shape

log_like2 = np.sum(log_like,axis=3)

print log_like2.shape

J_prior = GCE_calcs.analysis.get_J_prior_MC(J)

norm_test = np.trapz(J_prior,x=J)

print norm_test

J_prior = np.tile(J_prior[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))

norm_test2 = np.trapz(J_prior,x=J,axis=1)
print norm_test2.shape
print norm_test2

J_prior[J_prior==0]=1.e-300

post_nlp = log_like2 + np.log(J_prior)

post_test = np.exp(log_like2)*J_prior

post_test2 = np.trapz(post_test,x=J,axis=1)

post_test3 = np.log(post_test2)

post_test4 = post_test3 - post_test3.max()



post_nlp_min = np.amax(post_nlp,axis=1)


post_nlp_min2 = post_nlp_min - post_nlp_min.max()



levels = [0,1,3,6,12]
#plt.contour(mass_table,log_sigma,-post_log_pdf[:,2,:],levels)
#plt.contour(mass_table,np.log10(sigma),-post_nlp_min2,levels)
plt.contour(mass_table,np.log10(sigma),-post_test4,levels)
plt.savefig('array_test.png')





