import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt
from scipy import integrate

import GCE_calcs

GC_dens = GCE_calcs.calculations.density(0.5,0.5,20,1.2)

print GC_dens

GC_log_like = GCE_calcs.analysis.poisson_log_like(100.,95.)

print GC_log_like


print 'something'

k = np.array([2105.,1904.,1750.,1590.])
mu = np.array([2100.,1911.,1743.,1582.])

array_log_like = GCE_calcs.analysis.poisson_log_like(k,mu)

print array_log_like

