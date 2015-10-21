import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib import rc


rc('font',**{'family':'serif','serif':['Times New Roman']})

N1 = 100
N2 = 75

mass_table = 5.+.1*np.arange(N1)

csmin = -27.
csrange = 5.
  

MG_8 = np.loadtxt('output/approx2_tau_IC_8_20.txt') #importing the data
MG_6 = np.loadtxt('output/approx2_tau_IC_6_20.txt')
MG_10 = np.loadtxt('output/approx2_tau_IC_10_20.txt')
z_8 = np.zeros((N1,N2))
z_6 = np.zeros((N1,N2))
z_10 = np.zeros((N1,N2))
for i in range(N1):
	for j in range(N2):
		z_8[i][j] = MG_8[N2*i+j][2]
		z_6[i][j] = MG_6[N2*i+j][2]
		z_10[i][j] = MG_10[N2*i+j][2]

x = mass_table

y = pow(10,csmin+csrange*np.arange(N2)/N2)

levels = [0,1,3,6]# \Delta log-like  = (1,3,6)

plt.rcParams.update({'font.size': 28})
plt.contour(x, y, z_10.T,levels,colors ='r')
plt.contourf(x, y, z_8.T,levels,cmap=plt.cm.Blues_r,alpha=0.6)
plt.contour(x, y, z_6.T,levels,colors ='purple')
plt.xlabel('Mass (GeV)')
plt.xlim(5,15)
plt.ylim(1e-27,1e-23)
plt.ylabel(r'$\langle \sigma v \rangle$ (cm$^3$ sec$^{-1}$)')
plt.yscale('log')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.18)
plt.savefig('write-up/tau_data_IC.png')#plotting the data
plt.clf()

