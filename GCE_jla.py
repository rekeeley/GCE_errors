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

#J = np.array([1.e21,1.e22,1.e23])
#J = np.logspace(20.,24.,num=40)
J = np.array([1.5e22])

log_sigma = np.linspace(-27.,-23.,num=20)
#log_sigma = np.array([np.log10(3.e-26)])
#log_sigma = np.array([-23.,-24.,-25.,-26.,-27.])

#mass = np.array([10,20,30,40,50,60,70])

mu = GCE_calcs.calculations.get_mu(background,exposure, binned_spectra, J, log_sigma, mass_table)


plt.plot(bin_center,k)
plt.plot(bin_center,mu[10,0,35,:])
plt.savefig('spectra_test.png')

print k
print k - background

array_log_like = GCE_calcs.analysis.poisson_log_like(k,mu)

print array_log_like.shape

gauss_log_like = GCE_calcs.analysis.gauss_log_like_unnorm(k,mu)

print gauss_log_like.shape

n_cross, n_J, n_mass, n_spec = array_log_like.shape

J_prior = GCE_calcs.analysis.get_J_prior_MC(J)

J_prior = np.tile(J_prior,(n_cross, n_mass, n_spec, 1))

J_prior = J_prior.reshape(array_log_like.shape)

print J_prior.shape

post = gauss_log_like + np.log(J_prior)

print post.shape

post = np.sum(post,axis=3)

print post.shape

post_pdf = np.exp(post)

#J = np.tile()

post_pdf = np.trapz(post_pdf,x = J ,axis=1)

print post_pdf.shape

#print post_pdf

post_log_pdf = np.log(post_pdf)
post_log_pdf = post_log_pdf - post_log_pdf.max()

#print post_log_pdf

levels = [0,1.2,3.3,6.35,15]
plt.contour(mass_table,log_sigma,-post_log_pdf,levels)
plt.savefig('array_test.png')

