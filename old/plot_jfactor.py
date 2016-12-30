import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib import rc

#rc('font',**{'family':'serif','serif':['Times New Roman']})



jfactor1=np.loadtxt('output/j-factors/jfactor1_7.6830453226_0.28_1.12.txt')
jfactor2=np.loadtxt('output/j-factors/jfactor2_7.6830453226_0.28_1.12.txt')
jfactor3=np.loadtxt('output/j-factors/jfactor3_7.6830453226_0.28_1.12.txt')

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.weight': 100})
plt.rcParams.update({'font.family': 'serif'})



plt.plot(jfactor1[0,:],jfactor1[1,:]*1./jfactor1[1,:].max(),label = r'R$_s $ varied' )
plt.plot(jfactor2[0,:],jfactor2[1,:]*1./jfactor2[1,:].max(),label = r'R$_s$ and $\gamma$ varied')
plt.plot(jfactor3[0,:],jfactor3[1,:]*1./jfactor3[1,:].max(),label = r'R$_s$, $\gamma$ and $\rho_{local}$ varied' )

print 'aaaa'

plt.xscale('log')
plt.xlim(1e21,1e23)
plt.ylim(0,1.1)
plt.xlabel(r'J-factor (GeV cm$^{-5}$)',fontsize=28)
plt.ylabel('Scaled Likelihood',fontsize=28)
plt.legend(loc='best', prop = {'size':12})
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().subplots_adjust(left=0.18)
plt.savefig('write-up/plot_jfactor.pdf')
plt.clf()

