import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate




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

binned_spectra = np.loadtxt('spectra/binned/binned_spectra_'+output_file+'_'+str(dataset)+'_'+str(trunc)+'.dat')

def density(x,y,scale_radius,  gamma): #NFW density profile
	solar_radius = 8.25 #x is the los distance / solar radius  as to make the integral dimensionless
	R = np.sqrt(1.-2.*y*x + x*x)#R / solar radius
	return pow(R,-gamma)*pow((1 + R*solar_radius/scale_radius)/(1+solar_radius/scale_radius),gamma-3)

def J_factor(scale_radius,local_density,gamma):
	#integrating density^2 from x=0 to inf; x is line of sight distance/solarradius
	#theta = np.zeros((7,7))
	temp = np.zeros((7,7))
	for i in range(7):
		for j in range(7):
			y = np.cos(.5*(1+i+j)*np.pi/180.)
			integrand = lambda x: density(x,y,scale_radius,gamma)**2
			ans, err = integrate.quad(integrand, 0,np.inf)
			temp[i,j] = ans
	J = temp.mean()	
	kpctocm = 3.08568e21
	deltaomega = (7.*np.pi/180.)**2
	return  deltaomega*J*8.25*kpctocm*local_density*local_density
	
def spectra(m,c,sr,ld,g):
	mass = mass_table[m+N1_trunc]
	return data.T[7]+data.T[6]*binned_spectra[m+N1_trunc,:]*J_factor(sr,ld,g)*pow(10,c)/(8.*np.pi*mass*mass)





coeff = np.array([37.5153,-1.5093,1.63e-2,3.66e-4,-2.89237e-5,5.32e-7])
h = .67
Mmw = 1.5e12
rvir = 200.
argument = np.zeros(6)
for i in range(6):
	argument[i] = coeff[i]*pow(np.log(h*Mmw),i)
conc = np.sum(argument)

print conc

chi_out = np.zeros((N1,N2))
cnstrnt = np.array([conc,.23,1.12]) #center of gaussian for the marginalized parameters sr = scale radius; ld = local density; g = gamma
cnterr = np.array([.14,.06,.05])     #the std dev. of gaussian for the marginalized parameters

for i in range(N1):
	for j in range(N2):  
		print i, j									       	
		def chisquared(x): #the negative log-likelihood: 
			return -np.sum(data.T[5]*np.log(spectra(i,csmin+csrange*j/N2,x[0],x[1],x[2]))			       						-spectra(i,csmin+csrange*j/N2,x[0],x[1],x[2]) )  					       						+(np.log10(rvir/x[0])-np.log10(cnstrnt[0]))**2/(2*cnterr[0]**2) 					+(x[1]-cnstrnt[1])**2/(2*cnterr[1]**2)						      						+(x[2]-cnstrnt[2])**2/(2*cnterr[2]**2)
		x0 = np.array([rvir/conc,.23,1.12]) #initial guesses for the minimum of the log-likelihood in the marginalized space
		res = scop.minimize(chisquared, x0, method='nelder-mead',options={'ftol': 1e-10, 'disp': True,'maxfev': 5000}) #minimizing
		chi_out[i][j] = chisquared(res.x) 	
		#x = np.array([23.1,.3,1.2])	      
		#chi_out[i][j] = -np.sum(data.T[5]*np.log(spectra(i+1,csmin+csrange*j/N2,x[0],x[1],x[2]))			       					-spectra(i+1,csmin+csrange*j/N2,x[0],x[1],x[2]) )






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

np.savetxt('output/'+output_file+'_'+str(trunc)+'_'+str(dataset)+'.txt', data_out) #saving the data




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
plt.savefig('output/'+output_file+'_'+str(trunc)+'_'+str(dataset)+'.png')#plotting the data
plt.clf()







