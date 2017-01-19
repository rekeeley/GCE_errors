import numpy as np


#channel = 0  #tau
channel = 1 #bbar
#channel = 2 #SIDM

#model = 0  #noIC
#model = 1  #noB
model = 2  #IC data

dataset=20
trunc=8

mass_table = np.array([np.loadtxt('spectra/unbinned/tau/tau_mass_table.txt')[:,1],
			np.loadtxt('spectra/unbinned/bbar/bbar_mass_table.txt')[:,1],
            np.loadtxt('spectra/unbinned/SIDM/SIDM_mass_table.txt')[:,1]])[channel] #table of masses

N1 = len(mass_table)

file_path = np.array(['spectra/unbinned/tau/output-gammayield-',
                    'spectra/unbinned/bbar/muomega-gammayield-',
                    'spectra/unbinned/SIDM/muomega-gammayield-'])[channel]

output_file = np.array([['tau_full','tau_noMG','tau_IC'],
                        ['bbar_full','bbar_noMG','bbar_IC'],
                        ['SIDM_full','SIDM_noMG','SIDM_IC']])[channel][model]

output_path = np.array(['spectra/binned/tau/binned_spectra_',
                        'spectra/binned/bbar/binned_spectra_',
                        'spectra/binned/SIDM/binned_spectra_'])[channel]

raw= np.array([np.loadtxt('data/background/GCE_paper_fullmodel_spectrum.dat'),
		np.loadtxt('data/background/GCE_paper_noMG_spectrum.dat'),
		np.loadtxt('data/background/GCE_paper_IC_spectrum.dat')])[model]  

logdiff = np.zeros(29)
for i in range(29):
	logdiff[i] = -raw[i][0]+raw[i+1][0] #used to get the width of the bins


##### DATA #####

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

emin_GCE = 10**(raw[trunc:dataset,0] - 0.5*logdiff[0])
emax_GCE = 10**(raw[trunc:dataset,0] + 0.5*logdiff[0])

data_dwarfs = np.loadtxt('release-01-00-02/like_draco.txt')
emin = np.unique(data_dwarfs[:,0])/1000.
emax = np.unique(data_dwarfs[:,1])/1000. #delete copies and convert from GeV to MeV



##### BINNING FUNCTIONS #####

def spectra(number_spectra,emin,emax):
    binned_number_spectra = np.zeros(len(emin))
    for i in range(len(emin)):
        if emin[i] > number_spectra[-1,0]:
            imin = len(number_spectra[:,0])-1
            imax = len(number_spectra[:,0])-1
        elif emax[i] > number_spectra[-1,0]:
            imin = np.argmin(number_spectra[:,0] < emin[i])
            imax = len(number_spectra[:,0])-1
        else:
            imax = np.argmax(number_spectra[:,0] > emax[i])
            imin = np.argmin(number_spectra[:,0] < emin[i])
        binned_number_spectra[i] = np.trapz(number_spectra[imin:imax,1],x=number_spectra[imin:imax,0])
    return binned_number_spectra



def energy_spectra(number_spectra,emin,emax):
    binned_energy_spectra = np.zeros(len(emin))
    for i in range(len(emin)):
        if emin[i] > number_spectra[-1,0]:
            imin = len(number_spectra[:,0])-1
            imax = len(number_spectra[:,0])-1
        elif emax[i] > number_spectra[-1,0]:
            imin = np.argmin(number_spectra[:,0] < emin[i])
            imax = len(number_spectra[:,0])-1
        else:
            imax = np.argmax(number_spectra[:,0] > emax[i])
            imin = np.argmin(number_spectra[:,0] < emin[i])
        binned_energy_spectra[i] = np.trapz(number_spectra[imin:imax,0]*number_spectra[imin:imax,1],x=number_spectra[imin:imax,0])
    return binned_energy_spectra


##### DOING THE BINNING #####

data_out_GCE = np.zeros((N1,len(emin_GCE)))
for i in range(N1):
    file_number = str(i+1)
    number_spectra = np.loadtxt(file_path+file_number+'.dat')
    data_out_GCE[i,:] = spectra(number_spectra,emin_GCE,emax_GCE)

data_out_dwarf_energy = np.zeros((N1,len(emin)))
for i in range(N1):
    file_number = str(i+1)
    number_spectra = np.loadtxt(file_path+file_number+'.dat')
    data_out_dwarf_energy[i,:] = energy_spectra(number_spectra,emin,emax)


##### SAVING RESULTS #####

np.savetxt(output_path+output_file+'_'+str(dataset)+'_'+str(trunc)+'_GCE.txt',data_out_GCE )

np.savetxt(output_path+output_file+'_'+str(dataset)+'_'+str(trunc)+'dwarf_energy_spectra.txt',data_out_dwarf_energy)




