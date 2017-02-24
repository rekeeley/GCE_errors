import numpy as np


#channel = 0  #tau
channel = 1 #bbar
#channel = 2 #SIDM

#model = 0  #no x-bulge
model = 1  #x-bulge


mass_table = np.array([np.loadtxt('spectra/unbinned/tau/tau_mass_table.txt')[:,1],
			np.loadtxt('spectra/unbinned/bbar/bbar_mass_table.txt')[:,1],
            np.loadtxt('spectra/unbinned/SIDM/SIDM_mass_table.txt')[:,1]])[channel] #table of masses

N1 = len(mass_table)

file_path = np.array(['spectra/unbinned/tau/output-gammayield-',
                    'spectra/unbinned/bbar/muomega-gammayield-',
                    'spectra/unbinned/SIDM/muomega-gammayield-'])[channel]

output_file = np.array([['tau_noxbulge','tau_xbulge'],
                        ['bbar_noxbulge','bbar_xbulge'],
                        ['SIDM_noxbulge','SIDM_xbulge']])[channel][model]

output_path = np.array(['spectra/binned_0215/tau/binned_spectra_',
                        'spectra/binned_0215/bbar/binned_spectra_',
                        'spectra/binned_0215/SIDM/binned_spectra_'])[channel]

raw= np.array([np.loadtxt('data/background/GC_data_0215.dat'),
		np.loadtxt('data/background/GC_data_0215_with_stellar_templates.dat')])[model]




##### DATA #####
emin_GCE = raw[:,0]/1000.
emax_GCE = raw[:,1]/1000.

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

np.savetxt(output_path+output_file+'_GCE.txt',data_out_GCE )

np.savetxt(output_path+output_file+'_dwarf_energy_spectra.txt',data_out_dwarf_energy)




