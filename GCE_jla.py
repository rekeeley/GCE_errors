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


k =raw[trunc:dataset,5]

background = raw[trunc:dataset,7]

exposure = raw[trunc:dataset,6]

#print mass_table

#data = np.zeros((dataset-trunc,8))
#for i in range(trunc,dataset):
#    data[i-trunc][0] =  pow(10,raw[i][0]) #position (center) of bin
#    data[i-trunc][1] =  pow(10,raw[i][1]) #unused flux data
#    data[i-trunc][2] =  pow(10,raw[i][2]) #upper flux err
#    data[i-trunc][3] =  pow(10,raw[i][3]) #lower flux err
#    data[i-trunc][4] =  raw[i][4]
#    data[i-trunc][5] =  raw[i][5]#total number counts
#    data[i-trunc][6] =  raw[i][6]#exposure
#    data[i-trunc][7] =  raw[i][7]#background

binned_spectra = np.loadtxt('spectra/binned/binned_spectra_'+output_file+'_'+str(dataset)+'_'+str(trunc)+'.dat')


#k = np.array([2105.,1904.,1750.,1590.])
#mu = np.array([2100.,1911.,1743.,1582.])

#background = np.array([2105.,1904.,1750.,1590.])

#num_spec = np.array([1.2e-4,0.9e-4,0.8e-4,0.6e-4])

#J = np.array([1.e21,1.e22,1.e23])
J = np.logspace(20.,24.,num=40)

log_sigma = np.linspace(-27.,-23.,num=20)
#log_sigma = np.array([-23.,-24.,-25.,-26.,-27.])

#mass = np.array([10,20,30,40,50,60,70])

mu = GCE_calcs.calculations.get_mu(background,exposure, binned_spectra, J, log_sigma, mass_table)

array_log_like = GCE_calcs.analysis.poisson_log_like(k,mu)

print array_log_like.shape

J_prior = GCE_calcs.calculations.get_J_log_prior_fast(J)

J_prior = np.tile(J_prior,(12,20,50,1))

J_prior = J_prior.reshape(array_log_like.shape)

print J_prior.shape

post = array_log_like + J_prior

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

print post_log_pdf

levels = [0,1.2,3.3,6.35,15]
plt.contour(mass_table,log_sigma,-post_log_pdf,levels)
plt.savefig('array_test.png')

