import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate
import time


#channel = 0  #tau
channel = 1 #bbar

#model = 0  #noIC
#model = 1  #noB
model = 2  #IC data

dataset=20
trunc=6

#mass_table = np.array([np.loadtxt('spectra/tau/LSP-energies-original.dat')[:,1],
#			np.loadtxt('spectra/bbar/LSP-energies.dat')[:,1]])[channel] #table of masses 



file_path = np.array(['spectra/test/tau/muomega-gammayield-','spectra/test/bbar/muomega-gammayield-'])[channel]

output_file = np.array([['tau_full','tau_noMG','tau_IC'],['bbar_full','bbar_noMG','bbar_IC']])[channel][model]

#N1 = 120#number of points used in the mass axis

#mass_table = 20. + .5*np.arange(N1)

N1 = np.array([100,120])[channel]
mass_table = np.array([5.+.1*np.arange(N1),20. + .5*np.arange(N1)])[channel] #for tau case


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



def spectra(m):
    #m is the number of the PPPC4DMID file to load and integrate over
    #these different files are for the different spectra for the different masses
	file_number = str(m)
	number_spectra = np.loadtxt(file_path+file_number+'.dat')
	binned_number_spectra = np.zeros(len(data.T[0]))  #binning aka (integrating over the bins) the flux density
	for i in range(len(data.T[0])):
		l=0
		k=0
		if data[i][0]*pow(10,-logdiff[0]*.5) > number_spectra[len(number_spectra.T[0])-1][0]:
			binned_number_spectra[i] = 0
		elif data[i][0]*pow(10,logdiff[0]*.5) > number_spectra[len(number_spectra.T[0])-1][0]:	
			jmax = len(number_spectra.T[0])-1
			while number_spectra[k][0] < data[i][0]*pow(10,-logdiff[0]*.5):
				k=k+1
			jmin=k 
			binned_number_spectra[i]= np.trapz(number_spectra[jmin:jmax+1,1],x=number_spectra[jmin:jmax+1,0])		
		else: 
			while number_spectra[k][0] < data[i][0]*pow(10,-logdiff[0]*.5):
				k=k+1
			jmin=k 
			while number_spectra[l][0] < data[i][0]*pow(10,logdiff[0]*.5):
				l=l+1
			jmax=l
			binned_number_spectra[i]= np.trapz(number_spectra[jmin:jmax+1,1],x=number_spectra[jmin:jmax+1,0])
	return binned_number_spectra




data_out = np.zeros((N1,len(data)))


for i in range(N1):
	data_out[i,:] = spectra(i+1)


np.savetxt('spectra/test/binned/binned_spectra_'+output_file+'_'+str(dataset)+'_'+str(trunc)+'.dat', data_out)

data_dwarfs = np.loadtxt('release-01-00-02/like_draco.txt')
print data_dwarfs.shape
emin = np.unique(data_dwarfs[:,0])/1000.
emax = np.unique(data_dwarfs[:,1])/1000. #delete copies and convert from GeV to MeV

def spectra_dwarfs(m):
    file_number = str(m)
    number_spectra = np.loadtxt(file_path+file_number+'.dat')
    binned_number_spectra = np.zeros(len(emin))
    for i in range(len(emin)):
        if emin[i] > number_spectra[-1,0]:
            imin = len(number_spectra[:,0])-1
            imax = len(number_spectra[:,0])-1
            print 'the min of the draco energy bin is above the PPPC4DMID result'
        elif emax[i] > number_spectra[-1,0]:
            imin = np.argmin(number_spectra[:,0] < emin[i])
            imax = len(number_spectra[:,0])-1
            print 'the max of the draco energy bin is above the PPPC4DMID result'
        else:
            imax = np.argmax(number_spectra[:,0] > emax[i])
            imin = np.argmin(number_spectra[:,0] < emin[i])
        print 'the minima are...'
        print emin[i]
        print imin
        print number_spectra[imin,0]
        print 'the maxima are...'
        print emax[i]
        print imax
        print number_spectra[imax,0]
        binned_number_spectra[i] = np.trapz(number_spectra[imin:imax,1],x=number_spectra[imin:imax,0])
    return binned_number_spectra

data_out_draco = np.zeros((N1,len(emin)))
for i in range(N1):
    data_out_draco[i,:] = spectra_dwarfs(i+1)



np.savetxt('spectra/test/binned/binned_spectra_'+output_file+'_draco.txt',data_out_draco)

