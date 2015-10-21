readme

The first step is to take spectra from PPPC4DMID and integrate them over the bins of the data

There are 2 sets of spectra, one set that you gave me at the beginning of the project and another set that I made with PPPC4DMID.  The ones you gave me are in spectra/(bbar or tau) the ones I made with PPPC4DMID are in spectra/test/(bbar or tau)

There are 2 files that bin these spectra, one for your spectra and one for the PPPC4DMID spectra

Since there are (2 channels x [3 data models + 2 different data cuts] = 10) 10 cases there are 10 output binned spectra files (done entirely withe GCE_spectrabinning.py or GCE_specbin_PPPC.py)

GCE_analysis.py takes these integrated spectra and outputs contour data files in the output folder.

The various plot_X.py make the various plots, and puts them in the write-up folder

j-factor.py outputs monte carlo samples of the prior likelihood and bins them linearly or logarithmically and puts those data files in output/j-factors


