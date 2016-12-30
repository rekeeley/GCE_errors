import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib import rc


#rc('font',**{'family':'serif','serif':['Times New Roman']})

N1 = 120
N2 = 75

mass_table = 20. + .5*np.arange(N1)

csmin = -27.
csrange = 5.  

geringer = np.loadtxt('digitized_plots/ger-sam_bb.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
pass8 = np.loadtxt('digitized_plots/pass8.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
  

MG = np.loadtxt('output/approx2_bbar_full_8_20.txt') #importing the data
noMG = np.loadtxt('output/approx2_bbar_noMG_8_20.txt')
IC = np.loadtxt('output/approx2_bbar_IC_8_20.txt')#importing the data
extremeIC = np.loadtxt('output/extreme_bbar_IC_8_20.txt')

z_MG = np.zeros((N1,N2))
z_noMG = np.zeros((N1,N2))
z_IC = np.zeros((N1,N2))
extz_IC = np.zeros((N1,N2))
for i in range(N1):
	for j in range(N2):
		z_MG[i][j] = MG[N2*i+j][2]
		z_noMG[i][j] = noMG[N2*i+j][2]
		z_IC[i][j] = IC[N2*i+j][2]
		extz_IC[i][j] = extremeIC[N2*i+j][2]

x = mass_table

y = pow(10,csmin+csrange*np.ones(N2)*range(N2)/N2)



levels = [0,1,3,6]# \Delta log-like  = (1,3,6)

plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'font.weight': 100})
plt.rcParams.update({'font.family': 'serif'})

plt.contourf(x, y, z_noMG.T,levels,cmap=plt.cm.YlOrBr_r)
plt.contourf(x, y, z_IC.T,levels,cmap=plt.cm.Greens_r)
plt.contourf(x, y, z_MG.T,levels,cmap=plt.cm.Blues_r,alpha=.6)
plt.contourf(x, y, extz_IC.T,levels,cmap=plt.cm.Reds_r)
plt.plot(pass8[0,:],pow(10,pass8[1,:]),'k',label='Ackermann et al. (2015)')
#plt.plot(geringer[0,:],pow(10,geringer[1,:]),label = 'Geringer-Sameth et al. (2015)',color='0.75')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(right=.98)
plt.xlabel('Mass (GeV)')
plt.xlim(20,60)
plt.ylim(1e-27,1e-23)
plt.ylabel(r'$\langle \sigma v \rangle$ (cm$^3$ sec$^{-1}$)')
plt.yscale('log')
plt.gcf().subplots_adjust(bottom=0.19)
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(right=.94)
plt.legend(loc='upper left', prop = {'size':14})
plt.savefig('write-up/bbar_models.pdf')#plotting the data
plt.clf()



