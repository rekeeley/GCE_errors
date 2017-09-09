readme

The GCE_specbin_PPPC_0414.py files takes output files generated from PPPC4DMID and bins them over the energy bins defined by the data.  The model and used energy bins can be specified in this file.  

GCE_X.py takes these integrated spectra and outputs various statistics and plots.  X refers to the specific model.  The difference between the _profile files is the form of the data.  The data used by the _profile are log-likelihood profiles for an NFW component in each energy bin.  i.e. its a look up table where you supply the table with a number flux for a given energy bin had the table returns the log-likelihood for that number flux.  The files with out that suffix use a residual fitting analysis where the best fit number counts for the background (everything besides the NFW template) components of the GC fit are added to the exposure times the number flux for a given energy bin and this Poisson variable is compared to the observed number counts in that energy bin.  This latter technique underestimates the errors along the mass direction (for WIMPs) since it doesn’t marginalize over uncertainties in the backgrounds, just fixes the best fit values.

The spectra for the astrophysical models and the likelihood and prior functions are defined in the GCE_calcs folder in the calculations.py and analysis.py files respectively.

