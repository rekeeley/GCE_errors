readme

The first step is to take spectra from PPPC4DMID and integrate them over the bins of the data (GCE_specbin_PPPC.py)

The spectra were generated with PPPC4DMID and are located in in spectra/test/(bbar or tau)

Since there are (2 channels x [3 data models + 2 different data cuts] = 10) 10 cases there are 10 output binned spectra files 

GCE_analysis.py takes these integrated spectra and outputs contour data files in the output folder.

The various plot_X.py make the various plots, and puts them in the write-up folder

j-factor.py outputs monte carlo samples of the prior likelihood and bins them linearly or logarithmically and puts those data files in output/j-factors


