import matplotlib.pyplot as plt
import numpy as np



no_bulge_data = np.loadtxt('data/background/GC_data_0215.dat')

bulge_data = np.loadtxt('data/background/GC_data_0215_with_stellar_templates.dat')

emin_GCE = no_bulge_data[:,0]/1000.
emax_GCE = no_bulge_data[:,1]/1000.

bin_center = np.sqrt(emin_GCE*emax_GCE)
k = no_bulge_data[:,4]
background_nobulge = no_bulge_data[:,3]
background_bulge = bulge_data[:,3]
#exposure = raw[:,2]


plt.errorbar(bin_center, k, yerr=np.sqrt(k), color='c', label='Observed Counts')
plt.plot(bin_center, background_nobulge, 'm', label='No Bulge Background')
plt.plot(bin_center, background_bulge, 'y', label='X-Bulge Background')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.legend(loc='best')
plt.savefig('plot_0215_data.png')
plt.clf()


plt.errorbar(bin_center, k - background_nobulge, yerr=np.sqrt(k), color='m', label='No Bulge Residual')
plt.errorbar(bin_center, k - background_bulge, yerr=np.sqrt(k), color='y', label='Bulge Residual')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.legend(loc='best')
plt.savefig('plot_0215_data_residual.png')
plt.clf()
