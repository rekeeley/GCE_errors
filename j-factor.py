import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate
import time


Nsamples = 2000
resolution = 100

def density(x,scale_radius,  gamma): #NFW density profile
	solar_radius = 8.25
	R = np.sqrt(1 -2.*np.cos(2.0*np.pi/180.)*x + x*x)#R / solar radius
	return pow(R,-gamma)*pow((1 + R*solar_radius/scale_radius)/(1+solar_radius/scale_radius),gamma-3)

def J_factor(scale_radius,local_density,gamma):
	#integrating density^2 from x=0 to inf; x is line of sight distance
	integrand = lambda x: density(x,scale_radius,gamma)**2
	ans, err = integrate.quad(integrand, 0,np.inf)
	kpctocm = 3.08568e21
	deltaomega = (7.*np.pi/180.)**2
	return deltaomega*ans*8.25*kpctocm*local_density*local_density


coeff = np.array([37.5153,-1.5093,1.63e-2,3.66e-4,-2.89237e-5,5.32e-7])
h = .67
Mmw = 1.5e12
rvir = 200.
argument = np.zeros(6)
for i in range(6):
	argument[i] = coeff[i]*pow(np.log(h*Mmw),i)
conc = np.sum(argument)

#conc=15
#conc=25.
#conc=50.

#rholocal = 0.4
rholocal = 0.28
var_rho=.08

gamma = 1.12
var_gamma = .05

#gamma = 1.26
#var_gamma=.15

print conc



print J_factor(rvir/conc,rholocal,gamma)

data_quad = J_factor(rvir/conc,np.random.normal(rholocal,var_rho,1000000),gamma)

data1 = np.zeros(Nsamples)
data2 = np.zeros(Nsamples)
data3 = np.zeros(Nsamples)
for i in range(Nsamples):
	data1[i] = J_factor(rvir/np.random.lognormal(np.log(conc),.14,1),rholocal,gamma)
	data2[i] = J_factor(rvir/np.random.lognormal(np.log(conc),.14,1),rholocal,np.random.normal(gamma,var_gamma,1))
	data3[i] = J_factor(rvir/np.random.lognormal(np.log(conc),.14,1),
				np.random.normal(rholocal,var_rho,1),
				np.random.normal(gamma,var_gamma,1))


print 'done drawing samples'



bins1 = np.logspace(np.log10(data1.min()), np.log10(data1.max()), num=resolution)
bins2 = np.logspace(np.log10(data2.min()), np.log10(data2.max()), num=resolution)
bins3 = np.logspace(np.log10(data3.min()), np.log10(data3.max()), num=resolution)

bins_quad = np.linspace(data_quad.min(), data_quad.max(), num=resolution)
lin_bins1 = np.linspace(data1.min(), data1.max(), num=resolution)
lin_bins2 = np.linspace(data2.min(), data2.max(), num=resolution)
lin_bins3 = np.linspace(data3.min(), data3.max(), num=resolution)

bin_quad_pos = np.zeros(len(bins_quad)-1)
bin_positions1 = np.zeros(len(bins1)-1)
bin_positions2 = np.zeros(len(bins2)-1)
bin_positions3 = np.zeros(len(bins3)-1)


lin_bin_pos1 = np.zeros(len(bins1)-1)
lin_bin_pos2 = np.zeros(len(bins2)-1)
lin_bin_pos3 = np.zeros(len(bins3)-1)

for i in range(len(bins1)-1):
	bin_quad_pos[i] = pow(10,.5*(np.log10(bins_quad[i]) + np.log10(bins_quad[i+1])))
	bin_positions1[i] = pow(10,.5*(np.log10(bins1[i]) + np.log10(bins1[i+1])))
	bin_positions2[i] = pow(10,.5*(np.log10(bins2[i]) + np.log10(bins2[i+1])))
	bin_positions3[i] = pow(10,.5*(np.log10(bins3[i]) + np.log10(bins3[i+1])))
	lin_bin_pos1[i] = .5*(lin_bins1[i] + lin_bins1[i+1])
	lin_bin_pos2[i] = .5*(lin_bins2[i] + lin_bins2[i+1])
	lin_bin_pos3[i] = .5*(lin_bins3[i] + lin_bins3[i+1])



bins_means1 = np.histogram(data1,bins1)
bins_means2 = np.histogram(data2,bins2)
bins_means3 = np.histogram(data3,bins3)

bins_means_quad = np.histogram(data_quad,bins_quad)
lin_bins_means1 = np.histogram(data1,lin_bins1)
lin_bins_means2 = np.histogram(data2,lin_bins2)
lin_bins_means3 = np.histogram(data3,lin_bins3)



data_loglike = np.array([bin_positions3,-np.log(bins_means3[0]+.000001)+np.log(bins_means3[0].max())])


data_quad_out = np.array([bin_quad_pos,-np.log(bins_means_quad[0] + 1.e-10) + np.log(bins_means_quad[0].max())])
data_out1 = np.array([bin_positions1,bins_means1[0]])
data_out2 = np.array([bin_positions2,bins_means2[0]])
data_out3 = np.array([bin_positions3,bins_means3[0]])


lin_data_out1 = np.array([lin_bin_pos1,lin_bins_means1[0]])
lin_data_out2 = np.array([lin_bin_pos2,lin_bins_means2[0]])
lin_data_out3 = np.array([lin_bin_pos3,lin_bins_means3[0]])


np.savetxt('output/j-factors/jfactor1_'+str(conc)+'_'+str(rholocal)+'_'+str(gamma)+'.txt', data_out1) #saving the data
np.savetxt('output/j-factors/jfactor2_'+str(conc)+'_'+str(rholocal)+'_'+str(gamma)+'.txt', data_out2) #saving the data
np.savetxt('output/j-factors/jfactor3_'+str(conc)+'_'+str(rholocal)+'_'+str(gamma)+'.txt', data_out3) #saving the data

np.savetxt('output/j-factors/jfactor_loglike_'+str(conc)+'_'+str(rholocal)+'_'+str(gamma)+'.txt', data_loglike) #saving the data



np.savetxt('output/j-factors/jfactor_loglike.txt',data_quad_out)

np.savetxt('output/j-factors/jfactor1_lin_'+str(conc)+'_'+str(rholocal)+'_'+str(gamma)+'.txt', lin_data_out1) #saving the data
np.savetxt('output/j-factors/jfactor2_lin_'+str(conc)+'_'+str(rholocal)+'_'+str(gamma)+'.txt', lin_data_out2) #saving the data
np.savetxt('output/j-factors/jfactor3_lin_'+str(conc)+'_'+str(rholocal)+'_'+str(gamma)+'.txt', lin_data_out3) #saving the data





plt.plot(bin_positions1,bins_means1[0]*1./bins_means1[0].max(),label = r'R$_s$ varied')
plt.plot(bin_positions2,bins_means2[0]*1./bins_means2[0].max(),label = r'R$_s$ and $\gamma$ varied')
plt.plot(bin_positions3,bins_means3[0]*1./bins_means3[0].max(),label = r'R$_s$, $\gamma$ and $\rho_{local}$ varied')
plt.xscale('log')
plt.legend(loc='upper left', prop = {'size':14})
plt.savefig('output/j-factors/jfactorpdf_'+str(conc)+'_'+str(rholocal)+'_'+str(gamma)+'.png')
plt.clf()


plt.plot(lin_bin_pos1,lin_bins_means1[0]*1./lin_bins_means1[0].max(),label = r'R$_s$ varied')
plt.plot(lin_bin_pos2,lin_bins_means2[0]*1./lin_bins_means2[0].max(),label = r'R$_s$ and $\gamma$ varied')
plt.plot(lin_bin_pos3,lin_bins_means3[0]*1./lin_bins_means3[0].max(),label = r'R$_s$, $\gamma$ and $\rho_{local}$ varied')
plt.xscale('log')
plt.legend(loc='upper left', prop = {'size':14})
plt.savefig('output/j-factors/jfactor_lin_pdf_'+str(conc)+'_'+str(rholocal)+'_'+str(gamma)+'.png')
plt.clf()

