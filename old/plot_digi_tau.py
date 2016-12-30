import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib import rc

#rc('font',**{'family':'serif','serif':['Times New Roman']})

magic = np.loadtxt('digitized_plots/magictau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
#LAT = np.loadtxt('digitized_plots/fermilattau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
geringer = np.loadtxt('digitized_plots/ger-sam_tau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
pass8 = np.loadtxt('digitized_plots/pass8tau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
#HESS = np.loadtxt('digitized_plots/HESS.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
#gordon = np.loadtxt('digitized_plots/gordonv2.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
daylan = np.loadtxt('digitized_plots/daylantau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
calore = np.loadtxt('digitized_plots/caloretau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
relic = np.loadtxt('digitized_plots/relic.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)


N1 = 100
N2 = 75


mass_table = 5.+.1*np.arange(N1)


csmin = -27.
csrange = 5.

MG = np.loadtxt('output/approx2_tau_IC_8_20.txt') #saving the data
z = np.zeros((N1,N2))
for i in range(N1):
	for j in range(N2):
		z[i][j] = MG[N2*i+j][2]



x = mass_table



y = pow(10,csmin+csrange*np.arange(N2)/N2)

levels = [0,1.2,3.3,6.35,15]# \Delta log-like  = (1,3,6)

plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.weight': 100})
plt.rcParams.update({'font.family': 'serif'})

plt.contourf(x, y, z.T,levels,cmap=plt.cm.Greens_r)
plt.plot(pass8[0,:],pow(10,pass8[1,:]),'k',label='Ackermann et al. (2015)')
#plt.plot(LAT[0,:],pow(10,LAT[1,:]),label='Fermi-LAT MW Halo',color='0.75')
#plt.plot(geringer[0,:],pow(10,geringer[1,:]),label = 'Geringer-Sameth et al. (2015)',color='0.75')
#plt.plot(HESS[0,:],pow(10,HESS[1,:]),'r--',label='H.E.S.S GC Halo')
#plt.plot(magic[0,:],pow(10,magic[1,:]),color='orange',label='MAGIC Segue 1')
#plt.plot(gordon[0,:],pow(10,gordon[1,:]),label='Gordon & Macias 2013 (2$\sigma$)')
plt.plot(daylan[0,:],pow(10,daylan[1,:]),'r',label='Daylan et al. (2014) (2$\sigma$)')
plt.plot(calore[0,:],pow(10,calore[1,:]),'c',label='Calore et al. (2014) (2$\sigma$)')
plt.plot(relic[0,:],pow(10,relic[1,:]),'k--')
plt.text(15,5e-26,'Thermal Relic Cross Section',color='k',fontsize=10)
plt.text(15, 3.5e-26,'(Steigman et al. 2012)',color='k',fontsize=10)
plt.text(70, 1.5e-27,r'$\tau^{+} \tau^{-}$',fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-27,1e-23)
plt.xlim(5,1e2)
plt.xlabel('Mass (GeV)',fontsize=28)
plt.ylabel(r'$\langle \sigma v\rangle$ (cm$^3$ sec$^{-1}$)',fontsize=28)
plt.legend(loc='best', prop = {'size':14})
plt.gcf().subplots_adjust(bottom=0.19)
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(right=.94)
plt.savefig('write-up/plot_digi_tau.pdf')#plotting the data
plt.clf()
