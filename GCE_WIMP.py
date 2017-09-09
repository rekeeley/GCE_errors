import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
import argparse

import GCE_calcs


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--channel', type=int, default=0,
        help='annihilation channel: bbar=0 tau=1, default bbar')
    parser.add_argument('--model', type=int, default=0,
        help='background model used to generate the data used for the GCE: horiuchi et al=0, glliem=1, gal2yr=2')
    parser.add_argument('--trunc', type=int, default=0,
        help='number of data points to truncate at low energies, default 0')
    args = parser.parse_args()

    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams.update({'font.size': 24})

    channel = args.channel
    model = args.model
    trunc = args.trunc

###################
######  GCE part
###################


    channel_name = ['bbar','tau'][channel]
    model_name = ['kwa','glliem','gal2yr'][model]# corresponds to case A, B, C, D respectively
    mass_table = np.loadtxt('spectra/unbinned/'+channel_name+'/'+channel_name+'_mass_table.txt')[:,1]
    
    n_J=51
    J = np.logspace(19., 24., num=n_J, endpoint=True)

    n_mass = len(mass_table)
    mass_prior_norm = np.trapz(np.ones(n_mass), x=mass_table) #the inverse of this norm quantities are the prior on mass

    n_sigma = 61
    sigma = np.logspace(-29., -23., num=n_sigma, endpoint=True) #flat prior in logspace (scale-invariant)
    sigma_prior_norm = np.trapz(np.ones(n_sigma), x=np.log(sigma)) #the inverse of this norm quantities are the prior on sigma


##### DATA #####
    raw = np.loadtxt('data/background/GC_'+model_name+'.txt')
    raw = raw.T
    emin_GCE = raw[trunc:,0]/1000.
    emax_GCE = raw[trunc:,1]/1000.

    bin_center = np.sqrt(emin_GCE*emax_GCE)
    k = raw[trunc:,4]
    background = raw[trunc:,3]
    exposure = raw[trunc:,2]
    
##### ANALYSIS #####
    binned_spectra = np.loadtxt('spectra/binned_0414/'+channel_name+'/binned_spectra_'+channel_name+'_GCE.txt')[:,trunc:]

    mu = GCE_calcs.calculations.get_mu(background,exposure, binned_spectra, J, sigma, mass_table)

    k = np.tile(k,(n_sigma,n_J,n_mass,1))

    log_like_GCE_4d = GCE_calcs.analysis.poisson_log_like(k,mu) #a 4-d array of the log-likelihood with shape (n_sigma,n_J,n_mass,n_spec)

    log_like_GCE_3d = np.sum(log_like_GCE_4d,axis=3) #summing the log-like along the energy bin axis

    J_prior = np.loadtxt('data/J_factors/J_MC_'+model_name+'.txt')[1] #load the prior on the J-factor, calculated form a convolution of priors on rho_local, gamma, and the scale radius

    J_prior = np.tile(J_prior[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass)) #tiling to make the prior the same shape as the likelihood

    GCE_like_3d = np.exp(log_like_GCE_3d)*J_prior

    max_index_GCE = np.unravel_index(GCE_like_3d.argmax(),GCE_like_3d.shape)

    GCE_like_2d = np.trapz(GCE_like_3d, x=np.log10(J), axis=1) #marginalizing over the J factor
    
    evidence_GCE = np.trapz(np.trapz(GCE_like_2d,x = np.log(sigma),axis =0),x = mass_table,axis=0) / (sigma_prior_norm * mass_prior_norm)

########################
### PLOTS FOR GCE PART
########################


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
    plt.savefig('plots/WIMP/'+channel_name+'_'+model_name+'/residuals_GCE.pdf')
    plt.clf()

    np.savetxt('plots/WIMP/'+channel_name+'_'+model_name+'/residuals_GCE.txt', (bin_center, k[0,0,0,:], background, mu[max_index_GCE[0],max_index_GCE[1],max_index_GCE[2],:]))


    cmap = cm.cool
    levels = [0,1,3,6,10,15]
    manual_locations = [(41, 3e-26), (40, 1e-25), (39, 3e-25), (37, 1e-24), (34, 3e-24)]
    CS = plt.contour(mass_table,sigma,-np.log(GCE_like_2d) + np.log(GCE_like_2d.max()), levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
    plt.yscale('log')
    plt.xlabel('Mass [GeV]')
    plt.ylim(1e-28,1e-23)
    plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
    plt.title(r'GCE $-\Delta$Log-Likelihood Contours')
    plt.savefig('plots/WIMP/'+channel_name+'_'+model_name+'/GCE_contours.pdf')
    plt.clf()
    
    np.savetxt('plots/WIMP/'+channel_name+'_'+model_name+'/sigma_mass_posterior.txt', (-np.log(GCE_like_2d) + np.log(GCE_like_2d.max())))



################
### end GCE part
################


###################
### Dwarfs
###################

    like_name = ['like_bootes_I',
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
                'like_leo_V',
                'like_reticulum_II',
                'like_sculptor',
                'like_segue_1',
                'like_sextans',
                'like_ursa_major_I',
                'like_ursa_major_II',
                'like_ursa_minor',
                'like_willman_1']


    dwarf_mean_J = [18.2,
                    17.4,
                    17.6,
                    17.9,
                    19.0,
                    18.8,
                    17.8,
                    16.9,
                    17.8,
                    18.0,
                    16.3,
                    16.4,
                    18.9,
                    18.5,
                    19.4,
                    17.5,
                    17.9,
                    19.4,
                    18.9,
                    19.1]

    dwarf_var_J = [0.4,
                    0.3,
                    0.4,
                    0.1,
                    0.4,
                    0.1,
                    0.1,
                    0.7,
                    0.2,
                    0.2,
                    1.4,
                    0.9,
                    0.6,
                    0.1,
                    0.3,
                    0.2,
                    0.5,
                    0.4,
                    0.2,
                    0.3]


    binned_energy_spectra_dwarf =  np.loadtxt('spectra/binned_0414/'+channel_name+'/binned_spectra_'+channel_name+'_dwarf.txt')# this is a binned energy spectra (as opposed to number spectra)

    like_dwarf_2d = np.ones((n_sigma,n_mass))
    for i in range(len(like_name)):
        name = like_name[i]
        print name
        J_dwarf = np.logspace(dwarf_mean_J[i] - 4*dwarf_var_J[i],dwarf_mean_J[i]+4*dwarf_var_J[i],n_J)
        J_prior_dwarf = GCE_calcs.analysis.get_J_prior_dwarf(J_dwarf,dwarf_mean_J[i],dwarf_var_J[i])
        plt.plot(J_dwarf,J_prior_dwarf/J_prior_dwarf.max(),'c')
        plt.xscale('log')
        plt.ylim(0,1.1)
        plt.xlabel(r'J-factor [GeV$^2$ cm$^{-5}$]')
        plt.ylabel('Scaled Probability')
        plt.title('J-factor Likelihoods')
        plt.savefig('plots/WIMP/'+channel_name+'_'+model_name+'/'+name+'_J_factor_likelihoods.pdf')
        plt.clf()
        norm_test = np.trapz(J_prior_dwarf, x=np.log10(J_dwarf))
        assert abs(norm_test - 1)< 0.01, 'the normalization of the prior on the J-factor is off by more than 1%'
        J_prior_dwarf = np.tile(J_prior_dwarf[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass)) #tiling to make the prior of the J factor the same shape as the likelihood
        espec_dwarf = GCE_calcs.calculations.get_eflux(binned_energy_spectra_dwarf,J_dwarf,sigma,mass_table) #calculating the energy flux spectra
        log_like_dwarf_4d = GCE_calcs.analysis.dwarf_delta_log_like(espec_dwarf,name) #likelihood
        log_like_dwarf_3d = np.sum(log_like_dwarf_4d,axis=3) #summing over energy bins
        like_dwarf_3d = np.exp(log_like_dwarf_3d)*J_prior_dwarf
        like_ind_2d = np.trapz(like_dwarf_3d, x=np.log10(J_dwarf), axis=1) #marginalizing over J factor
        CS = plt.contour(mass_table,sigma,-np.log(like_ind_2d) + np.log(like_ind_2d.max()),levels)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.yscale('log')
        plt.xlabel('Mass [GeV]')
        plt.ylabel('Cross Section [cm^3 sec^-1]')
        plt.savefig('plots/WIMP/'+channel_name+'_'+model_name+'/'+name+'_contours.png')
        plt.clf()
        like_dwarf_2d *= like_ind_2d


    CS = plt.contour(mass_table, sigma, -np.log(like_dwarf_2d/like_dwarf_2d.max()), levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.yscale('log')
    plt.xlabel('Mass [GeV]')
    plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
    plt.ylim(1e-28,1e-23)
    plt.title(r'Combined Dwarf $-\Delta$Log-Likelihood Contours')
    plt.savefig('plots/WIMP/'+channel_name+'_'+model_name+'/dwarf_contours.pdf')
    plt.clf()
    
    np.savetxt('plots/WIMP/'+channel_name+'_'+model_name+'/dwarf_contours.txt',-np.log(like_dwarf_2d/like_dwarf_2d.max()))

    like_dwarf_1d = np.trapz(like_dwarf_2d, x=mass_table, axis=1) #marginalizing over the mass
    like_GCE_1d = np.trapz(GCE_like_2d, x=mass_table, axis=1) #marginalizing over the mass

    plt.subplots_adjust(left=0.12, bottom=0.14, right=0.96, top=0.96)
    plt.plot(sigma, like_dwarf_1d/like_dwarf_1d.max(), 'c', label='Combined Dwarfs', linewidth=2.0)
    plt.plot(sigma, like_GCE_1d/like_GCE_1d.max(), 'm', label='GCE', linewidth=2.0)
    plt.xscale('log')
    plt.xlabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
    plt.ylabel('Scaled Probabilty')
    plt.ylim(0,1.1)
    plt.legend(loc='upper right', frameon=False, fontsize=22)
    plt.savefig('plots/WIMP/'+channel_name+'_'+model_name+'/cross_section_posteriors.pdf')
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
    plt.savefig('plots/WIMP/'+channel_name+'_'+model_name+'/combo_contours.png')
    plt.clf()

    np.savetxt('plots/WIMP/'+channel_name+'_'+model_name+'/cross_section_posteriors.txt',(sigma,like_dwarf_1d/like_dwarf_1d.max(),like_GCE_1d/like_GCE_1d.max()))

    evidence_combo = np.trapz(np.trapz(combo_like_2d, x=np.log(sigma), axis=0), x=mass_table, axis=0)/ (sigma_prior_norm * mass_prior_norm)


    print 'the GCE evidence is ' + str(evidence_GCE)
    print 'the dwarf evidence is ' +str(evidence_dwarf)
    print 'the product of the dwarf and GCE evidence is ' + str(evidence_dwarf*evidence_GCE)
    print 'the combined evidence is ' +str(evidence_combo)

    np.savetxt('plots/WIMP/'+channel_name+'_'+model_name+'/evidences.txt',(evidence_GCE, evidence_dwarf, evidence_dwarf*evidence_GCE, evidence_combo, evidence_dwarf*evidence_GCE/evidence_combo, GCE_like_3d.max()) )

if __name__ == '__main__':
    main()