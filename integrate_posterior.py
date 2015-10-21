import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib import rc
import math as mt


N1 = 120
N2 = 75

MG = np.loadtxt('output/approx2_bbar_IC_8_20.txt') #saving the data
z = np.zeros((N1,N2))
for i in range(N1):
	for j in range(N2):
		z[i][j] = MG[N2*i+j][2]

pdf = np.exp(-z)

norm = pdf.sum()

norm_pdf = pdf/norm

cum_1=0.
for i in range(N1):
	for j in range(N2):
		if z[i][j]<1.:
			cum_1+=norm_pdf[i][j]

cum_3=0.
for i in range(N1):
	for j in range(N2):
		if z[i][j]<3.:
			cum_3+=norm_pdf[i][j]

cum_6=0.
for i in range(N1):
	for j in range(N2):
		if z[i][j]<6.:
			cum_6+=norm_pdf[i][j]

cum_14=0.
for i in range(N1):
	for j in range(N2):
		if z[i][j]<14.:
			cum_14+=norm_pdf[i][j]

cumulative = np.zeros(30)		
for k in range(len(cumulative)):
	for i in range(N1):
		for j in range(N2):
			if z[i][j]<.5*(k+1):
				cumulative[k]+=norm_pdf[i][j]

print cum_1, mt.erf(1/np.sqrt(2))
print cum_3, mt.erf(2/np.sqrt(2))
print cum_6, mt.erf(3/np.sqrt(2))
print cum_14, mt.erf(5/np.sqrt(2))

print cumulative

plt.plot(.5*(np.arange(30)+1),1-cumulative,label = 'integrated posterior')
plt.plot(.5*(np.arange(30)+1),np.ones(30)*(1-mt.erf(1/np.sqrt(2))),label = '1 sigma')
plt.plot(.5*(np.arange(30)+1),np.ones(30)*(1-mt.erf(2/np.sqrt(2))),label = '2 sigma')
plt.plot(.5*(np.arange(30)+1),np.ones(30)*(1-mt.erf(3/np.sqrt(2))),label ='3 sigma' )
plt.plot(.5*(np.arange(30)+1),np.ones(30)*(1-mt.erf(5/np.sqrt(2))),label = '5 sigma')
plt.yscale('log')
plt.legend(loc='lower left', prop = {'size':14})
plt.xlim(6,6.5)
plt.xlabel(r'$\Delta$Log-Likelihood Contour',fontsize=18)
plt.ylabel('Probability Outside Contour',fontsize=18)
plt.savefig('posterior.png')

