import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
from scipy import integrate

import GCE_calcs

rc('font',**{'family':'serif','serif':['Times New Roman']})
plt.rcParams.update({'font.size': 24})

channel = 0  #tau
#channel = 1 #bbar

model = 0  #no x-bulge
#model = 1  #x-bulge

###################
######  GCE part
###################



mass_table = np.array([np.loadtxt('spectra/unbinned/tau_limits/tau_mass_table.txt')[:,1],
                       np.loadtxt('spectra/unbinned/bbar_limits/bbar_mass_table.txt')[:,1]])[channel] #table of masses

#mass_table = np.array([])[channel]

file_path = np.array(['spectra/binned_0414/tau_limits/binned_spectra_',
                        'spectra/binned_0414/bbar_limits/binned_spectra_'])[channel]

file_name = np.array([['tau_xbulge','tau_xbulge'],
                        ['bbar_xbulge','bbar_xbulge']])[channel][model]


n_J=201
J = np.logspace(19.,23.,num=n_J,endpoint=True)

n_mass = len(mass_table)

n_sigma = 101
sigma = np.logspace(-29.,-24.,num=n_sigma, endpoint=True)

sigma_prior_norm = np.trapz(np.ones(n_sigma),x = np.log(sigma))# a flat prior in linear space for mass and logspace for cross section

mass_prior_norm = np.trapz(np.ones(n_mass),x = mass_table)

raw= np.array([np.loadtxt('data/background/GC_data_0414.txt'),
		np.loadtxt('data/background/GC_data_xbulge_0414.txt')])[model]




##### DATA #####
emin_GCE = raw[:,0]/1000.
emax_GCE = raw[:,1]/1000.

bin_center = np.sqrt(emin_GCE*emax_GCE)
k = raw[:,4]
background = raw[:,4]
exposure = raw[:,2]

binned_spectra = np.loadtxt(file_path+file_name+'_GCE.txt')

mu = GCE_calcs.calculations.get_mu(background,exposure, binned_spectra, J, sigma, mass_table)

k = np.tile(k,(n_sigma,n_J,n_mass,1))

log_like_GCE_4d = GCE_calcs.analysis.poisson_log_like(k,mu) #a 4-d array of the log-likelihood with shape (n_sigma,n_J,n_mass,n_spec)

log_like_GCE_3d = np.sum(log_like_GCE_4d,axis=3) #summing the log-like along the energy bin axis

J_prior = GCE_calcs.analysis.get_J_prior_MC(J)

norm_test = np.trapz(J_prior,x=J)
assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the J-factor is off by more than 1%'

plt.plot(J,J_prior/J_prior.max(),'c',label = 'Zhang et al 2012')
plt.xscale('log')
plt.ylim(0,1.1)
plt.xlabel(r'J-factor [GeV$^2$ cm$^{-5}$]')
plt.ylabel('Scaled Probability')
plt.legend(loc='best')
plt.title('J-factor Likelihoods')
plt.savefig('plots/WIMP_limits/0414/J_factor_likelihoods.pdf')
plt.clf()


J_prior = np.tile(J_prior[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))

GCE_like_3d = np.exp(log_like_GCE_3d)*J_prior

max_index_GCE = np.unravel_index(GCE_like_3d.argmax(),GCE_like_3d.shape)
print GCE_like_3d.max()

plt.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.96)
plt.errorbar(bin_center, k[0,0,0,:]-background, yerr=np.sqrt(k[0,0,0,:]), color='c', label='Observed Residual', linewidth=2.0)
plt.plot(bin_center, mu[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2],:]-background, 'm', label='Expected Residual', linewidth=2.0)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.2,1e2)
plt.ylim(1e1,1e5)
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.legend(loc='upper right', frameon=False, fontsize=22)
plt.savefig('plots/WIMP_limits/0414/'+file_name+'_test_residuals_GCE.pdf')
plt.clf()

GCE_like_2d = np.trapz(GCE_like_3d, x=J, axis=1)

cmap = cm.cool
levels = [0,1,3,6,10,15]
#manual_locations = [(41, 3e-26), (40, 1e-25), (39, 3e-25), (37, 1e-24), (34, 3e-24)]
CS = plt.contour(mass_table,sigma,-np.log(GCE_like_2d) + np.log(GCE_like_2d.max()), levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
#plt.clabel(CS, inline=0, fontsize=10, fmt='%1.f', manual=manual_locations)
#plt.text(41, 3e-26, r'1$\sigma$', color='k', fontsize=10)
#plt.text(40, 1e-25, r'2$\sigma$', color='k', fontsize=10)
#plt.text(39, 3e-25, r'3$\sigma$', color='k', fontsize=10)
#plt.text(37, 1e-24, r'4$\sigma$', color='k', fontsize=10)
#plt.text(35, 3e-24, r'5$\sigma$', color='k', fontsize=10)
plt.subplots_adjust(left=0.14, bottom=0.14, right=0.97, top=0.96)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Mass [GeV]')
plt.xlim(10,500)
plt.ylim(sigma[0],sigma[-1])
plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
#plt.title(r'GCE $-\Delta$Log-Likelihood Contours')
plt.savefig('plots/WIMP_limits/0414/'+file_name+'_GCE_contours.pdf')
plt.clf()

plt.subplots_adjust(left=0.14, bottom=0.14, right=0.97, top=0.96)
CS = plt.contour(mass_table,sigma,-np.log(GCE_like_2d) + np.log(GCE_like_2d.max()), levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Mass [GeV]')
plt.xlim(30,70)
plt.ylim(sigma[0],sigma[-1])
plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
#plt.title(r'GCE $-\Delta$Log-Likelihood Contours')
plt.savefig('plots/WIMP_limits/0414/'+file_name+'_GCE_contours_zoom.pdf')
plt.clf()


evidence_GCE = np.trapz(np.trapz(GCE_like_2d,x = np.log(sigma),axis =0),x = mass_table,axis=0) / (sigma_prior_norm * mass_prior_norm)

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


binned_energy_spectra_dwarf =  np.loadtxt(file_path+file_name+'_dwarf_energy_spectra.txt')

like_dwarf_2d = np.ones((n_sigma,n_mass))
for i in range(len(like_name)):
    name = like_name[i]
    print name
    J_dwarf = np.logspace(dwarf_mean_J[i] - 5*dwarf_var_J[i],dwarf_mean_J[i]+5*dwarf_var_J[i],n_J)
    J_prior_dwarf = GCE_calcs.analysis.get_J_prior_dwarf(J_dwarf,dwarf_mean_J[i],dwarf_var_J[i])
    norm_test = np.trapz(J_prior_dwarf, x=J_dwarf)
    assert abs(norm_test - 1) < 0.01 , 'the normalization of the prior on the J-factor is off by more than 1%'
    J_prior_dwarf = np.tile(J_prior_dwarf[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass))
    espec_dwarf = GCE_calcs.calculations.get_eflux(binned_energy_spectra_dwarf,J_dwarf,sigma,mass_table)
    log_like_dwarf_4d = GCE_calcs.analysis.dwarf_delta_log_like(espec_dwarf,name)
    log_like_dwarf_3d = np.sum(log_like_dwarf_4d,axis=3)
    like_dwarf_3d = np.exp(log_like_dwarf_3d)*J_prior_dwarf
    like_ind_2d = np.trapz(like_dwarf_3d,x=J_dwarf,axis=1)
    CS = plt.contour(mass_table,sigma,-np.log(like_ind_2d) + np.log(like_ind_2d.max()),levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.subplots_adjust(left=0.14, bottom=0.14, right=0.97, top=0.96)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Cross Section [cm^3 sec^-1]')
    plt.savefig('plots/WIMP_limits/0414/'+file_name+'_'+name+'_contours.png')
    plt.clf()
    like_dwarf_2d *= like_ind_2d


CS = plt.contour(mass_table, sigma, -np.log(like_dwarf_2d/like_dwarf_2d.max()), levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
plt.clabel(CS, inline=1, fontsize=10)
plt.yscale('log')
plt.xscale('log')
plt.xlim(10,500)
plt.xlabel('Mass [GeV]')
plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
plt.ylim(sigma[0],sigma[-1])
plt.title(r'Combined Dwarf $-\Delta$Log-Likelihood Contours')
plt.savefig('plots/WIMP_limits/0414/'+file_name+'_dwarf_contours.pdf')
plt.clf()


like_dwarf_1d = np.trapz(like_dwarf_2d, x=mass_table, axis=1)
like_GCE_1d = np.trapz(GCE_like_2d, x=mass_table, axis=1)

plt.subplots_adjust(left=0.12, bottom=0.14, right=0.96, top=0.96)
plt.plot(sigma, like_dwarf_1d/like_dwarf_1d.max(), 'c', label='Combined Dwarfs', linewidth=2.0)
plt.plot(sigma, like_GCE_1d/like_GCE_1d.max(), 'm', label='GCE', linewidth=2.0)
plt.xscale('log')
plt.xlabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
plt.ylabel('Scaled Probabilty')
plt.ylim(0,1.1)
plt.legend(loc='upper right', frameon=False, fontsize=22)
#plt.title('Cross Section Posteriors')
plt.savefig('plots/WIMP_limits/0414/'+file_name+'_cross_section_posteriors.pdf')
plt.clf()

evidence_dwarf = np.trapz(np.trapz(like_dwarf_2d, x=np.log(sigma), axis=0), x=mass_table, axis=0)/ (sigma_prior_norm * mass_prior_norm)

####################
####### C-C-C-COMBO
####################

combo_like_2d = like_dwarf_2d  * GCE_like_2d

CS = plt.contour(mass_table,sigma,-np.log(combo_like_2d/combo_like_2d.max()),levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
plt.clabel(CS, inline=1, fontsize=10)
plt.subplots_adjust(left=0.14, bottom=0.14, right=0.98, top=0.96)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Mass [GeV]')
plt.xlim(10,500)
plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
plt.savefig('plots/WIMP_limits/0414/'+file_name+'_combo_contours.pdf')
plt.clf()



#Fermi_pass_8_digi_bbar = [[6,10,25],[5e-27,5e-27,8e-27]]
#Fermi_pass_8_digi_tau = [[6],[5e-27]]

if channel==1:
    magic = np.loadtxt('digitized_plots/magic.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    #LAT = np.loadtxt('digitized_plots/LAT.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    geringer = np.loadtxt('digitized_plots/ger-sam_bb.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    pass8 = np.loadtxt('digitized_plots/pass8.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    HESS = np.loadtxt('digitized_plots/HESS.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    gordon = np.loadtxt('digitized_plots/gordonv2.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    daylan = np.loadtxt('digitized_plots/daylanv2.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    calore = np.loadtxt('digitized_plots/calorev2.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    relic = np.loadtxt('digitized_plots/relic.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    pass8_tau = np.loadtxt('digitized_plots/pass8tau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    #HESS = np.loadtxt('digitized_plots/HESS.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    #gordon = np.loadtxt('digitized_plots/gordonv2.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    daylan_tau = np.loadtxt('digitized_plots/daylantau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    calore_tai = np.loadtxt('digitized_plots/caloretau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
else:
    magic = np.loadtxt('digitized_plots/magictau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    #LAT = np.loadtxt('digitized_plots/fermilattau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    geringer = np.loadtxt('digitized_plots/ger-sam_tau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    pass8 = np.loadtxt('digitized_plots/pass8tau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    #HESS = np.loadtxt('digitized_plots/HESS.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    #gordon = np.loadtxt('digitized_plots/gordonv2.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    gordon = np.loadtxt('digitized_plots/gordonv2.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    daylan = np.loadtxt('digitized_plots/daylantau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    calore = np.loadtxt('digitized_plots/caloretau.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)
    relic = np.loadtxt('digitized_plots/relic.csv',delimiter=',',unpack=True,dtype=float,skiprows=6)


levels_95 = [0,2.]
CS = plt.contour(mass_table, sigma, -np.log(combo_like_2d/combo_like_2d.max()), levels_95, colors='c',linewidth=2.0)
plt.plot(pass8[0,:],pow(10,pass8[1,:]),'m',label='Pass 8 Combined dSphs',linewidth=2.0)
#plt.clabel(CS, inline=1, fontsize=10)
plt.subplots_adjust(left=0.15, bottom=0.14, right=0.98, top=0.96)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Mass [GeV]')
plt.xlim(10,500)
plt.ylim(sigma[0],sigma[-1])
plt.legend(loc='upper right', frameon=False, fontsize=22)
plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
plt.savefig('plots/WIMP_limits/0414/'+file_name+'_combo_contours_95_percent_clean.pdf')
plt.clf()


levels_95 = [0,2.]
CS = plt.contour(mass_table, sigma, -np.log(combo_like_2d/combo_like_2d.max()), levels_95, colors='c', linewidth=2.0)
plt.plot(pass8[0,:],pow(10,pass8[1,:]),'m',label='Pass 8 Combined dSphs',linewidth=2.0)
plt.plot(gordon[0,:],pow(10,gordon[1,:]),label='Gordon & Macias 2013 (2$\sigma$)',linewidth=2.0)
plt.plot(daylan[0,:],pow(10,daylan[1,:]),label='Daylan et al. 2014 (2$\sigma$)',linewidth=2.0)
plt.plot(calore[0,:],pow(10,calore[1,:]),label='Calore et al. 2014 (2$\sigma$)',linewidth=2.0)
plt.plot(relic[0,:],pow(10,relic[1,:]),'k--',linewidth=2.0)
#plt.clabel(CS, inline=1, fontsize=10)
plt.subplots_adjust(left=0.15, bottom=0.14, right=0.98, top=0.96)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Mass [GeV]')
plt.xlim(10,500)
plt.ylim(1e-28,1e-23)
plt.legend(loc='upper right', frameon=False, fontsize=22)
plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
plt.savefig('plots/WIMP_limits/0414/'+file_name+'_combo_contours_95_percent.pdf')
plt.clf()



evidence_combo = np.trapz(np.trapz(combo_like_2d ,x = np.log(sigma),axis=0),x =mass_table,axis=0)/ (sigma_prior_norm * mass_prior_norm)


print 'the GCE evidence is ' + str(evidence_GCE)
print 'the dwarf evidence is ' +str(evidence_dwarf)
print 'the product of the dwarf and GCE evidence is ' + str(evidence_dwarf*evidence_GCE)
print 'the combined evidence is ' +str(evidence_combo)

np.savetxt(file_name+'_evidences.txt',np.array([evidence_GCE, evidence_dwarf, evidence_dwarf*evidence_GCE, evidence_combo, evidence_dwarf*evidence_GCE/evidence_combo, GCE_like_3d.max()]) )


#dloglike_combo = combo_like_2d/combo_like_2d.max()

#dloglike_combo = np.zeros((n_sigma+1,n_mass))

#dloglike_combo[0,:] = np.ones(n_mass)

#for i in range(n_mass):
#    for j in range(n_sigma):
#        dloglike_combo[j+1,i] = combo_like_2d/combo_like_2d.max()[j,i]


