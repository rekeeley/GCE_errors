import numpy as np


#channel = 0  #tau
#channel = 1 #bbar
channel = 2 #SIDM
#channel = 3 #bbar_limits
#channel = 4 #tau_limits




mass_table = [np.loadtxt('spectra/unbinned/tau/tau_mass_table.txt')[:,1],
			np.loadtxt('spectra/unbinned/bbar/bbar_mass_table.txt')[:,1],
            np.loadtxt('spectra/unbinned/SIDM/SIDM_mass_table.txt')[:,1],
            np.loadtxt('spectra/unbinned/bbar_limits/bbar_mass_table.txt')[:,1],
            np.loadtxt('spectra/unbinned/tau_limits/tau_mass_table.txt')[:,1]][channel] #table of masses

N1 = len(mass_table)

file_path = ['spectra/unbinned/tau/output-gammayield-',
                'spectra/unbinned/bbar/output-gammayield-',
                'spectra/unbinned/SIDM/output-gammayield-',
                'spectra/unbinned/bbar_limits/output-gammayield-',
                'spectra/unbinned/bbar_limits/output-gammayield-'][channel]

output_file = ['tau','bbar','SIDM','bbar_limits','tau_limits'][channel]

output_path = ['spectra/binned_0414/tau/binned_spectra_',
                'spectra/binned_0414/bbar/binned_spectra_',
                'spectra/binned_0414/SIDM/binned_spectra_',
                    'spectra/binned_0414/bbar_limits/binned_spectra_',
                    'spectra/binned_0414/tau_limits/binned_spectra_'][channel]

raw= np.loadtxt('data/background/GC_data_wen.txt')


raw_old = np.loadtxt('data/background/GCE_paper_fullmodel_spectrum.dat')

loge = raw_old[:,0]
dloge = loge[1]-loge[0]

emin_old = 10**(loge-0.5*dloge)
emax_old = 10**(loge+0.5*dloge)

##### DATA #####
emin_GCE = raw[:,0]/1000.
emax_GCE = raw[:,1]/1000.

data_dwarfs = np.loadtxt('release-01-00-00/like_draco.txt')
emin = np.unique(data_dwarfs[:,0])/1000.
emax = np.unique(data_dwarfs[:,1])/1000. #delete copies and convert from GeV to MeV

e_MIT = 2*np.logspace(-1,3,41)
emin_MIT = e_MIT[:-1]
emax_MIT = e_MIT[1:]

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

data_out_MIT = np.zeros((N1,len(emin_MIT)))
for i in range(N1):
    file_number = str(i+1)
    number_spectra = np.loadtxt(file_path+file_number+'.dat')
    data_out_MIT[i,:] = spectra(number_spectra,emin_MIT,emax_MIT)

data_out_old = np.zeros((N1,len(emin_old)))
for i in range(N1):
    file_number = str(i+1)
    number_spectra = np.loadtxt(file_path+file_number+'.dat')
    data_out_old[i,:] = spectra(number_spectra,emin_old,emax_old)

##### SAVING RESULTS #####

print data_out_GCE.shape

np.savetxt(output_path+output_file+'_GCE.txt',data_out_GCE )

np.savetxt(output_path+output_file+'_dwarf.txt',data_out_dwarf_energy)

np.savetxt(output_path+output_file+'_MIT.txt',data_out_MIT)

np.savetxt(output_path+output_file+'_old.txt',data_out_old)




