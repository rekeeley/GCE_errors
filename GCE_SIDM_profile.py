import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
import argparse

import GCE_calcs

# Parse command-line arguments.
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=int, default=0,
    help='background model used to generate the data used for the GCE: horiuchi et al=0, glliem=1, gal2yr=2')
parser.add_argument('--trunc', type=int, default=0,
    help='number of data points to truncate at low energies, default 0')
args = parser.parse_args()

rc('font',**{'family':'serif','serif':['Times New Roman']})
plt.rcParams.update({'font.size': 24})


model = args.model
trunc = args.trunc

###################
######  GCE part
###################


model_name = ['kwa','glliem','gal2yr','MIT'][model]# corresponds to case A, B, C, D respectively
mass_table = np.loadtxt('spectra/unbinned/SIDM/SIDM_mass_table.txt')[:,1]
file_name = model_name+'_profile'

n_J=51
norm_range = np.logspace(19.,24.,n_J,endpoint=True)

n_mass = len(mass_table)
mass_prior_norm = np.trapz(np.ones(n_mass), x=mass_table)

n_sigma = 61
sigma = np.logspace(-29.,-23.,num=n_sigma,endpoint=True)
sigma_prior_norm = np.trapz(np.ones(n_sigma), x=np.log(sigma))
# a flat prior in linear space for mass and logspace for cross section


##### DATA #####
loaded = np.load('data/background/GC_'+model_name+'.npz')
emin = loaded['emin'][trunc:,0]
emax = loaded['emax'][trunc:,0]
flux_max = np.zeros(len(emin))
bin_center = np.sqrt(emin*emax)
exposure = loaded['exposure'][trunc:]
loglike = -loaded['LL'][trunc:]# the loaded quantities are actual negative log probabilites so thats why theres a negative
exposure = np.tile(exposure[:,np.newaxis],(1,loglike.shape[1]))
nflux = loaded['counts'][trunc:]/exposure
    
for i in range(len(flux_max)):
    flux_max[i] = nflux[i,loglike[i,:].argmax()]


##### ANALYSIS #####
if model_name=='MIT':
    binned_spectra = np.loadtxt('spectra/binned_0414/SIDM/binned_spectra_SIDM_MIT.txt')[:,trunc:]
else:
    binned_spectra = np.loadtxt('spectra/binned_0414/SIDM/binned_spectra_SIDM_GCE.txt')[:,trunc:]#difference due to different energy bins

nspec_GCE = GCE_calcs.calculations.get_eflux(binned_spectra,norm_range,sigma,mass_table) #it says get e_flux but since a binned number flux (dN/dE) is input, it returns a number spectra

log_like_GCE_4d = GCE_calcs.analysis.GCE_delta_log_like(nspec_GCE,model_name,trunc)

log_like_GCE_3d = np.sum(log_like_GCE_4d,axis=3) #summing the log-like along the energy bin axis

GC_SM_mu = 2.6e8
GC_SM_sigma = 0.1*GC_SM_mu
norm_prior = GCE_calcs.analysis.get_SIDM_prior(norm_range, GC_SM_mu, GC_SM_sigma)

max_norm_index = norm_prior.argmax()

norm_test = np.trapz(norm_prior, x=norm_range)
assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the J-factor is off by more than 1%'

norm_prior = np.tile(norm_prior[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))

GCE_like_3d = np.exp(log_like_GCE_3d)*norm_prior

GCE_like_2d = np.trapz(GCE_like_3d, x=norm_range, axis=1)

max_index_GCE = np.unravel_index(GCE_like_3d.argmax(),GCE_like_3d.shape)

evidence_GCE = np.trapz(np.trapz(GCE_like_2d, x=np.log(sigma), axis=0), x=mass_table, axis=0) / (sigma_prior_norm * mass_prior_norm)


########################
### PLOTS FOR GCE PART
########################
plt.plot(norm_range,norm_prior[0,:,0]/norm_prior[0,:,0].max(),'c')
plt.xscale('log')
plt.ylim(0,1.1)
plt.xlabel('Normalization')
plt.ylabel('Scaled Probability')
plt.title('Normalization Likelihoods')
plt.savefig('plots/SIDM/'+file_name+'/norm_likelihoods.png')
plt.clf()


plt.plot(bin_center, flux_max, label='Best fit flux')
plt.plot(bin_center, nspec_GCE[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2],:],label = str(mass_table[max_index_GCE[2]]))
plt.plot(bin_center, nspec_GCE[max_index_GCE[0]+1,max_index_GCE[1],max_index_GCE[2],:],label = str(mass_table[max_index_GCE[2]]))
plt.plot(bin_center, nspec_GCE[max_index_GCE[0]-1,max_index_GCE[1],max_index_GCE[2],:],label = str(mass_table[max_index_GCE[2]]))
plt.plot(bin_center, nspec_GCE[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2]+10,:],label = str(mass_table[max_index_GCE[2]]))
plt.plot(bin_center, nspec_GCE[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2]-10,:],label = str(mass_table[max_index_GCE[2]-5]))
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-10,1e-7)
plt.legend(loc='upper right', frameon=False, fontsize=22)
plt.savefig('plots/SIDM/'+file_name+'/fluxes.png')
plt.clf()

cmap = cm.cool
levels = [0,1,3,6,10,15]
manual_locations = [(41, 3e-26), (40, 1e-25), (39, 3e-25), (37, 1e-24), (34, 3e-24)]
CS = plt.contour(mass_table,sigma,-np.log(GCE_like_2d) + np.log(GCE_like_2d.max()), levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
plt.yscale('log')
plt.xlabel('Mass [GeV]')
plt.ylim(sigma[0],sigma[-1])
plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
plt.title(r'GCE $-\Delta$Log-Likelihood Contours')
plt.savefig('plots/SIDM/'+file_name+'/GCE_contours.pdf')
plt.clf()



################
### end GCE part
################


###################
### Dwarfs
###################


like_name = np.array(['like_bootes_I',
                        'like_canes_venatici_I',
                        'like_canes_venatici_II',
                        'like_carina',
                        'like_coma_berenices',
                        'like_draco',
                        'like_fornax',
                        'like_hercules',
                        'like_leo_I',
                        'like_leo_II',
                        'like_leo_IV',
                        'like_sculptor',
                        'like_segue_1',
                        'like_sextans',
                        'like_ursa_major_I',
                        'like_ursa_major_II',
                        'like_ursa_minor',
                        'like_willman_1'])


dwarf_mean_J = np.array([18.8,
                        17.7,
                        17.9,
                        18.1,
                        19.0,
                        18.8,
                        18.2,
                        18.1,
                        17.7,
                        17.6,
                        17.9,
                        18.6,
                        19.5,
                        18.4,
                        18.3,
                        19.3,
                        18.8,
                        19.1])

dwarf_var_J = np.array([0.22,
                        0.26,
                        0.25,
                        0.23,
                        0.25,
                        0.16,
                        0.21,
                        0.25,
                        0.18,
                        0.18,
                        0.28,
                        0.18,
                        0.29,
                        0.27,
                        0.24,
                        0.28,
                        0.19,
                        0.31])

#arxiv.org/pdf/1204.1562.pdf

dwarf_SM_mu = 1.e6*np.array([ 0.029,
                            0.23,
                            0.0079,
                            0.38,
                            0.0037,
                            0.29,
                            20,
                            0.037,
                            5.5,
                            0.74,
                            0.019,
                            2.3,
                            0.00034,
                            0.44,
                            0.014,
                            0.0041,
                            0.29,
                            0.0010])




binned_energy_spectra_dwarf =  np.loadtxt('spectra/binned_0414/SIDM/binned_spectra_SIDM_dwarf.txt')

like_dwarf_2d = np.ones((n_sigma,n_mass))
for i in range(len(like_name)):
    name = like_name[i]
    print name
    dwarf_norm_mean = dwarf_mean_J[i] + np.log10(dwarf_SM_mu[i]/GC_SM_mu)
    norm_dwarf = np.logspace(dwarf_norm_mean-5*dwarf_var_J[i], dwarf_norm_mean+5*dwarf_var_J[i], n_J)
    norm_prior_dwarf = GCE_calcs.analysis.get_SIDM_dwarf_prior(norm_dwarf, dwarf_mean_J[i], dwarf_var_J[i],dwarf_SM_mu[i]/GC_SM_mu, 0.1*dwarf_SM_mu[i]/GC_SM_mu)
    norm_test = np.trapz(norm_prior_dwarf, x=norm_dwarf)
    assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the J-factor is off by more than 1%'
    plt.plot(norm_dwarf,norm_prior_dwarf/norm_prior_dwarf.max(),'c')
    plt.xscale('log')
    plt.ylim(0,1.1)
    plt.xlabel('Normalization')
    plt.ylabel('Scaled Probability')
    plt.title('Normalization Likelihoods')
    plt.savefig('plots/SIDM/'+file_name+'/'+name+'_norm_likelihoods.png')
    plt.clf()
    norm_prior_dwarf = np.tile(norm_prior_dwarf[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))
    espec_dwarf = GCE_calcs.calculations.get_eflux(binned_energy_spectra_dwarf, norm_dwarf, sigma, mass_table)
    log_like_dwarf_4d = GCE_calcs.analysis.dwarf_delta_log_like(espec_dwarf,name)
    log_like_dwarf_3d = np.sum(log_like_dwarf_4d,axis=3)
    like_dwarf_3d = np.exp(log_like_dwarf_3d)*norm_prior_dwarf
    like_ind_2d = np.trapz(like_dwarf_3d,x=norm_dwarf,axis=1)
    CS = plt.contour(mass_table,sigma,-np.log(like_ind_2d) + np.log(like_ind_2d.max()),levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.yscale('log')
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Cross Section [cm^3 sec^-1]')
    plt.savefig('plots/SIDM/'+file_name+'/'+name+'_contours.png')
    plt.clf()
    like_dwarf_2d *= like_ind_2d


CS = plt.contour(mass_table, sigma, -np.log(like_dwarf_2d/like_dwarf_2d.max()), levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
plt.clabel(CS, inline=1, fontsize=10)
plt.yscale('log')
plt.xlabel('Mass [GeV]')
plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
plt.ylim(1e-28,1e-23)
plt.title(r'Combined Dwarf $-\Delta$Log-Likelihood Contours')
plt.savefig('plots/SIDM/'+file_name+'/dwarf_contours.pdf')
plt.clf()


like_dwarf_1d = np.trapz(like_dwarf_2d, x=mass_table, axis=1)
like_GCE_1d = np.trapz(GCE_like_2d, x=mass_table, axis=1)

plt.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.96)
plt.plot(sigma, like_dwarf_1d/like_dwarf_1d.max(), 'c', label='Combined Dwarfs', linewidth=2.0)
plt.plot(sigma, like_GCE_1d/like_GCE_1d.max(), 'm', label='GCE', linewidth=2.0)
plt.xscale('log')
plt.xlabel(r'Normalization')
plt.ylabel('Scaled Probabilty')
plt.ylim(0,1.1)
plt.legend(loc='upper right', frameon=False, fontsize=22)
plt.savefig('plots/SIDM/'+file_name+'/cross_section_posteriors.pdf')
plt.clf()

evidence_dwarf = np.trapz(np.trapz(like_dwarf_2d, x=np.log(sigma), axis=0), x=mass_table, axis=0)/ (sigma_prior_norm * mass_prior_norm)


####################
####### C-C-C-COMBO
####################

combo_like_2d = like_dwarf_2d  * GCE_like_2d

CS = plt.contour(mass_table,sigma,-np.log(combo_like_2d/combo_like_2d.max()),levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.yscale('log')
plt.xlabel('Mass [GeV]')
plt.ylabel('Cross Section [cm^3 sec^-1]')
plt.savefig('plots/SIDM/'+file_name+'/combo_contours.png')
plt.clf()


evidence_combo = np.trapz(np.trapz(combo_like_2d , x=np.log(sigma), axis=0), x=mass_table,axis=0)/ (sigma_prior_norm * mass_prior_norm)

print 'the max likelihood for the GCE part is '+str(GCE_like_3d.max())
print 'the GCE evidence is ' + str(evidence_GCE)
print 'the dwarf evidence is ' +str(evidence_dwarf)
print 'the product of the dwarf and GCE evidence is ' + str(evidence_dwarf*evidence_GCE)
print 'the combined evidence is ' +str(evidence_combo)

np.savetxt('plots/SIDM/'+file_name+'/SIDM_evidences.txt',np.array([evidence_GCE, evidence_dwarf, evidence_dwarf*evidence_GCE, evidence_combo, evidence_dwarf*evidence_GCE/evidence_combo, GCE_like_3d.max()]) )
