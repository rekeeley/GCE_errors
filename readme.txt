readme

The GCE_specbin_PPPC.py files takes output files generated from PPPC4DMID and bins them over the energy bins defined by the data.  The model and used energy bins can be specified in this file.  

GCE_jla_X.py takes these integrated spectra and outputs various statistics and plots.  X refers to the specific model.

The spectra for the astrophysical models and the likelihood and prior functions are defined in the GCE_calcs folder in the calculations.py and analysis.py files respectively.

J_factor_plot.py outputs monte carlo KDE samples of the prior likelihood for different information about the local dark matter density and plots them.  It also makes a plot of the MW dark matter halo with a standard and contracted scale radii. 

