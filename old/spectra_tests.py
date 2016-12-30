import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate


#channel = 0  #tau
channel = 1 #bbar

#model = 0  #MG aka full
#model = 1  #noMG
model = 2  #IC data

###################
######  GCE part
###################

trunc = 8  #how many data points truncated from the front
dataset=20 #largest data point

mass_table = np.array([np.loadtxt('spectra/unbinned/tau/tau_mass_table.txt')[:,1],
                       np.loadtxt('spectra/unbinned/bbar/bbar_mass_table.txt')[:,1]])[channel] #table of masses

#mass_table = np.array([])[channel]

file_path = np.array(['spectra/binned/tau/binned_spectra_','spectra/binned/bbar/binned_spectra_'])[channel]

file_name = np.array([['tau_full','tau_noMG','tau_IC'],['bbar_full','bbar_noMG','bbar_IC']])[channel][model]

binned_spectra = np.loadtxt(file_path+file_name+'_'+str(dataset)+'_'+str(trunc)+'_GCE.txt')
