import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from scipy import array





#channel = 0  #tau
channel = 1 #bbar

#model = 0  #noIC (no IC)
#model = 1  #noB (no Bremsstrahlung / no IC)
model = 2  #IC data

trunc = 8  #how many data points truncated from the low energy end
dataset=20 #highest energy data point (30 max)

file_path = np.array(['spectra/tau/output-gammayield-','spectra/bbar/muomega-gammayield-'])[channel]

output_file = np.array([['tau_full','tau_noMG','tau_IC'],['bbar_full','bbar_noMG','bbar_IC']])[channel][model]

N1 = np.array([100,120])[channel]  #number of points in the mass axis
mass_table = np.array([5.+.1*np.arange(N1),20. + .5*np.arange(N1)])[channel] 


N2 = 75 #number of points used in the cross-section axis
csmin = -27.
csrange = 5.  #fit the order of magnitude of the cross-section so 1e-27 to 1e-22

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

binned_spectra = np.loadtxt('spectra/test/binned/binned_spectra_'+output_file+'_'+str(dataset)+'_'+str(trunc)+'.dat')

Jfactor = np.loadtxt('output/j-factors/jfactor_loglike.txt')  #loading the file that has the log-likelihood of the J-factor

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

jloglike =  interp1d(np.log10(Jfactor[0,:]),Jfactor[1,:],kind='linear')

jloglikeextra = extrap1d(jloglike)

exponent = np.arange(22.5,24.33,.01)

exponent2 = np.arange(10,26,.01)

#plt.plot(pow(10,exponent),jloglike(pow(10,exponent)),label='cubic')

plt.plot(pow(10,exponent2),jloglikeextra(exponent2),label='extrapolated')
#plt.plot(pow(10,exponent),jloglike(exponent),label='interpolation')
plt.plot(Jfactor[0,:],Jfactor[1,:],label='data')
plt.legend(loc='best', prop = {'size':14})
plt.xscale('log')
plt.ylim(0,20)
plt.savefig('inter_test.png')
plt.clf()

def spectra(m,c,jfactor):
	mass = mass_table[m]
	return data.T[7]+data.T[6]*binned_spectra[m,:]*pow(10,jfactor)*pow(10,c)/(8.*np.pi*mass*mass)

chi_out = np.zeros((N1,N2))
 #center of gaussian for the marginalized parameters sr = scale radius; ld = local density; g = gamma
     #the std dev. of gaussian for the marginalized parameters	

for i in range(N1):
	for j in range(N2):  
		print i,j									       	
		def chisquared(x): #the negative log-likelihood: 
			return -np.sum(data.T[5]*np.log(spectra(i,csmin+csrange*j/N2,x[0]))			       						-spectra(i,csmin+csrange*j/N2,x[0]) )  					       				 		+jloglikeextra([x[0]])						      		
		x0 = np.array([23.]) #initial guesses for the minimum of the log-likelihood in the marginalized space
		res = scop.minimize(chisquared, x0, method='nelder-mead',options={'ftol': 1e-10, 'disp': True,'maxfev': 10000,'maxiter':5000}) #minimizing
		chi_out[i][j] = chisquared(res.x) 	



xgnu=np.zeros(N1*N2)
ygnu=np.zeros(N1*N2)
zgnu=np.zeros(N1*N2)
data_out = np.zeros((N1*N2,3))

for i in range(N1):
	for j in range(N2):
		xgnu[N2*i+j]=mass_table[i]
		ygnu[N2*i+j]= csmin+csrange*j/N2
		zgnu[N2*i+j] = chi_out[i][j]- chi_out.min()


data_out.T[0] = xgnu
data_out.T[1] = ygnu
data_out.T[2] = zgnu

np.savetxt('output/interp_'+output_file+'_'+str(trunc)+'_'+str(dataset)+'.txt', data_out) #saving the data


x = np.zeros(N1)
for i in range(N1):
	x[i] = mass_table[i]

y = csmin+csrange*np.ones(N2)*range(N2)/N2
z= chi_out.T - chi_out.min() 

levels = [0,1,3,6]# \Delta log-like  = (1,3,6)

CS = plt.contourf(x, y, z,levels,cmap=plt.cm.Blues_r)



plt.xlabel('Mass [GeV]')
plt.ylabel('Cross Section [cm^3 sec^-1]')
plt.savefig('output/interp_'+output_file+'_'+str(trunc)+'_'+str(dataset)+'.png')#plotting the data
plt.clf()







