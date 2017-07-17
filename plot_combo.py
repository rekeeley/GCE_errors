import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
from scipy import integrate

import GCE_calcs

rc('font',**{'family':'serif','serif':['Times New Roman']})
plt.rcParams.update({'font.size': 24})



bbar_best_fit = np.loadtxt('plots/WIMP/bbar_wen/residuals_GCE.txt')
tau_best_fit = np.loadtxt('plots/WIMP/tau_wen/residuals_GCE.txt')
lp_best_fit = np.loadtxt('plots/log_parabola_zoa/wen/residuals_GCE.txt')
ec_best_fit = np.loadtxt('plots/exp_cutoff/wen/residuals_GCE.txt')
SIDM_best_fit = np.loadtxt('plots/SIDM/wen/residuals_GCE.txt')

bin_center = bbar_best_fit[0]
k = bbar_best_fit[1]
background = bbar_best_fit[2]

mu_bbar = bbar_best_fit[3]
mu_tau = tau_best_fit[3]
mu_lp = lp_best_fit[3]
mu_ec = ec_best_fit[3]
mu_SIDM = SIDM_best_fit[3]




plt.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.96)
plt.errorbar(bin_center, k-background, yerr=np.sqrt(k), color='k', label='Data', linewidth=2.0)
plt.plot(bin_center, mu_bbar-background, 'r', label=r'$b\bar{b}$', linewidth=2.0)
plt.plot(bin_center, mu_tau-background, 'g', label=r'$\tau$', linewidth=2.0)
plt.plot(bin_center, mu_lp-background, 'b', label=r'Log-Parab', linewidth=2.0)
plt.plot(bin_center, mu_ec-background, 'y', label=r'Exp Cutoff', linewidth=2.0)
plt.plot(bin_center, mu_SIDM-background, 'c', label=r'SIDM', linewidth=2.0)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.2,1e2)
plt.ylim(1e1,1e5)
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.legend(loc='upper right', frameon=False, fontsize=22)
plt.savefig('plots/combo_residuals_GCE_wen.pdf')
plt.clf()



bbar_best_fit = np.loadtxt('plots/WIMP/bbar_gll_iem/residuals_GCE.txt')
tau_best_fit = np.loadtxt('plots/WIMP/tau_gll_iem/residuals_GCE.txt')
lp_best_fit = np.loadtxt('plots/log_parabola_zoa/gll_iem/residuals_GCE.txt')
ec_best_fit = np.loadtxt('plots/exp_cutoff/gll_iem/residuals_GCE.txt')
SIDM_best_fit = np.loadtxt('plots/SIDM/gll_iem/residuals_GCE.txt')

bin_center = bbar_best_fit[0]
k = bbar_best_fit[1]
background = bbar_best_fit[2]

mu_bbar = bbar_best_fit[3]
mu_tau = tau_best_fit[3]
mu_lp = lp_best_fit[3]
mu_ec = ec_best_fit[3]
mu_SIDM = SIDM_best_fit[3]


plt.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.96)
plt.errorbar(bin_center, k-background, yerr=np.sqrt(k), color='k', label='Data', linewidth=2.0)
plt.plot(bin_center, mu_bbar-background, 'r', label=r'$b\bar{b}$', linewidth=2.0)
plt.plot(bin_center, mu_tau-background, 'g', label=r'$\tau$', linewidth=2.0)
plt.plot(bin_center, mu_lp-background, 'b', label=r'Log-Parab', linewidth=2.0)
plt.plot(bin_center, mu_ec-background, 'y', label=r'Exp Cutoff', linewidth=2.0)
plt.plot(bin_center, mu_SIDM-background, 'c', label=r'SIDM', linewidth=2.0)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.2,1e2)
plt.ylim(1e1,1e5)
plt.xlabel('Energy [GeV]')
plt.ylabel('Number Counts')
plt.legend(loc='upper right', frameon=False, fontsize=22)
plt.savefig('plots/combo_residuals_GCE_gll_iem.pdf')
plt.clf()
