import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate




channel = 0  #tau
#channel = 1 #bbar

model = 0  #MG aka full
#model = 1  #noMG
#model = 2  #IC data

trunc = 10  #how many data points truncated from the front
dataset=20 #largest data point

#mass_table = np.array([np.loadtxt('spectra/tau/LSP-energies-original.dat')[:,1],
#			np.loadtxt('spectra/bbar/LSP-energies.dat')[:,1]])[channel] #table of masses 

file_path = np.array(['spectra/tau/output-gammayield-','spectra/bbar/muomega-gammayield-'])[channel]

output_file = np.array([['tau_full','tau_noMG','tau_IC'],['bbar_full','bbar_noMG','bbar_IC']])[channel][model]

N1 = 90
mass_table = 5.+.5*np.arange(N1) #for tau case

N1_trunc=0

#N1 = 120#number of points used in the mass axis
#mass_table = 20. + .5*np.arange(N1)

N2 = 75 #number of points used in the cross-section axis
csmin = -27.
csrange = 5.  #fit the order of magnitude of the cross-section so 1e-26 to 1e-23

cs_table = csmin + csrange*np.arange(N2)/N2

raw= np.array([np.loadtxt('data/background/GCE_paper_fullmodel_spectrum.dat'),
		np.loadtxt('data/background/GCE_paper_noMG_spectrum.dat'),
		np.loadtxt('data/background/GCE_paper_IC_spectrum.dat')])[model]  

logdiff = np.zeros(29)
for i in range(29):
	logdiff[i] = -raw[i][0]+raw[i+1][0] #used to get the width of the bins

data = np.zeros((dataset-trunc,8))
for i in range(trunc,dataset):
	data[i-trunc][0] =  pow(10,raw[i][0]) #position (center) of bin
	data[i-trunc][1] =  pow(10,raw[i][1]) #unused flux data
	data[i-trunc][2] =  pow(10,raw[i][2]) #upper flux err
	data[i-trunc][3] =  pow(10,raw[i][3]) #lower flux err
	data[i-trunc][4] =  raw[i][4]
	data[i-trunc][5] =  raw[i][5]#total number counts
	data[i-trunc][6] =  raw[i][6]#exposure
	data[i-trunc][7] =  raw[i][7]#background

binned_spectra = np.loadtxt('spectra/test/binned/binned_spectra_'+output_file+'_'+str(dataset)+'_'+str(trunc)+'.dat')

coeff = np.array([37.5153,-1.5093,1.63e-2,3.66e-4,-2.89237e-5,5.32e-7])
h = .67
Mmw = 1.5e12#try another mass .5 e12
rvir = 200.
argument = np.zeros(6)
for i in range(6):
	argument[i] = coeff[i]*pow(np.log(h*Mmw),i)
conc = np.sum(argument)

print conc

def density(x): #NFW density profile
	scale_radius = rvir/conc
	gamma = 1.12
	solar_radius = 8.25 #x is the los distance / solar radius  as to make the integral dimensionless
	R = np.sqrt(1.-2.*np.cos(7.*np.pi/180.)*x + x*x)#R / solar radius
	return pow(R,-gamma)*pow((1 + R*solar_radius/scale_radius)/(1+solar_radius/scale_radius),gamma-3)

integrand = lambda x: density(x)**2
ans, err = integrate.quad(integrand, 0,np.inf)

print ans
print err

def J_factor(local_density):
	#integrating density^2 from x=0 to inf; x is line of sight distance/solarradius
	kpctocm = 3.08568e21
	return ans*8.25*kpctocm*local_density*local_density
	
def spectra(m,c,J):
	mass = mass_table[m]
	return data.T[7]+data.T[6]*binned_spectra[m]*J*pow(10,c)/(8.*np.pi*mass*mass)

def post_like(m,c,J):
	return -np.sum(data.T[5]*np.log(spectra(m,c,J))-spectra(m,c,J))

int_res=200
terms = np.zeros(int_res)
for i in range(len(terms)):
	terms[i] = post_like(40,-26.,pow(10, 22.8+.55*i/int_res ))

terms = terms-terms.min()

print terms

print pow(10, 22.8+.55*np.arange(int_res)/int_res) 

correction = np.trapz(np.exp(-terms),pow(10, 22.8+.55*np.arange(int_res)/int_res ))

print np.exp(-terms)

print correction

print np.log(correction)

chi_out = np.zeros((N1,N2))
cnstrnt = np.array([conc,.28,1.12]) #center of gaussian for the marginalized parameters sr = scale radius; ld = local density; g = gamma
cnterr = np.array([.14,.08,.05])     #the std dev. of gaussian for the marginalized parameters

J_table = 21.+4.*np.arange(int_res)/int_res

print J_table

print cs_table

def J_factor_prior(J):
	return 3.e-46*(J - 3.e23)**2

for i in range(N1):
	print i
	for j in range(N2):
		cs = cs_table[j]
		for k in range(int_res):
			J = pow(10,J_table[k])
			terms[k] = post_like(i,cs,J)*J_factor_prior(J)
		base = terms.min()
		terms = terms-terms.min()
		chi_out[i][j] = np.log(np.trapz(np.exp(-terms),pow(10,J_table)))

print chi_out.min()

print chi_out.max()
		
xgnu=np.zeros(N1*N2)
ygnu=np.zeros(N1*N2)
zgnu=np.zeros(N1*N2)
data_out = np.zeros((N1*N2,3))

for i in range(N1):
	for j in range(N2):
		xgnu[N2*i+j]=mass_table[i+N1_trunc]
		ygnu[N2*i+j]= csmin+csrange*j/N2
		zgnu[N2*i+j] = chi_out[i][j]- chi_out.min()


data_out.T[0] = xgnu
data_out.T[1] = ygnu
data_out.T[2] = zgnu

np.savetxt('output/integrate_'+output_file+'_'+str(trunc)+'_'+str(dataset)+'.txt', data_out) #saving the data




x = np.zeros(N1)
for i in range(N1):
	x[i] = mass_table[i+N1_trunc]

y = csmin+csrange*np.ones(N2)*range(N2)/N2
z= chi_out.T - chi_out.min() 

levels = [0,1,3,6]# \Delta log-like  = (1,3,6)

CS = plt.contourf(x, y, z,levels,cmap=plt.cm.Blues_r)



# Label every level using strings


plt.xlabel('Mass [GeV]')
plt.ylabel('Cross Section [cm^3 sec^-1]')
plt.savefig('output/integrate_'+output_file+'_'+str(trunc)+'_'+str(dataset)+'.png')#plotting the data
plt.clf()







