import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib import rc


rc('font',**{'family':'serif','serif':['Times New Roman']})

N1 = 120
N2 = 75

mass_table = 20. + .5*np.arange(N1)

csmin = -27.
csrange = 5.  
  

geringer = np.loadtxt('digitized_plots/ger-sam_bb.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
pass8 = np.loadtxt('digitized_plots/pass8.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)

MG_8 = np.loadtxt('output/approx2_bbar_IC_8_20.txt') #importing the data
MG_6 = np.loadtxt('output/approx2_bbar_IC_6_20.txt')
MG_10 = np.loadtxt('output/approx2_bbar_IC_10_20.txt')
extremeIC = np.loadtxt('output/extreme_bbar_IC_8_20.txt')
z_8 = np.zeros((N1,N2))
z_6 = np.zeros((N1,N2))
z_10 = np.zeros((N1,N2))
extz_IC = np.zeros((N1,N2))
for i in range(N1):
	for j in range(N2):
		z_8[i][j] = MG_8[N2*i+j][2]
		z_6[i][j] = MG_6[N2*i+j][2]
		z_10[i][j] = MG_10[N2*i+j][2]
		extz_IC[i][j] = extremeIC[N2*i+j][2]

x = mass_table

y = pow(10,csmin+csrange*np.ones(N2)*range(N2)/N2)

levels = [0,1,3,6]# \Delta log-like  = (1,3,6)

plt.rcParams.update({'font.size': 28})
plt.contour(x, y, z_10.T,levels,colors ='r')
plt.contourf(x, y, z_8.T,levels,cmap=plt.cm.Blues_r,alpha=0.6)
plt.contour(x, y, z_6.T,levels,colors = 'purple')
#plt.contourf(x, y, extz_IC.T,levels,cmap=plt.cm.Reds_r)
#plt.plot(pass8[0,:],pow(10,pass8[1,:]),'k',label='Pass 8 Combined dSphs')
#plt.plot(geringer[0,:],pow(10,geringer[1,:]),label = 'Geringer-Sameth et al. 2015',color='0.75')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.18)
#plt.gca().tight_layout()
plt.xlabel('Mass (GeV)')
plt.xlim(20,80)
plt.ylim(1e-27,1e-23)
plt.ylabel(r'$\langle \sigma v \rangle$ (cm$^3$ sec$^{-1}$)')
plt.yscale('log')
plt.savefig('write-up/bbar_data_IC.png')#plotting the data
plt.clf()

