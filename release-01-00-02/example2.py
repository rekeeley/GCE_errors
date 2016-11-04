#!/usr/bin/env python
#
# example2.py
#
# This example demonstrates how to load
# and plot the bin-by-bin likelihood limits.
#
# 
# Copyright (C) 2013 Alex Drlica-Wagner <kadrlica@fnal.gov>
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from matplotlib import rc
rc('text', usetex=True)

import numpy as np
import pylab as plt

# Load the bin-by-bin likelihood limits
infile = 'bin_by_bin_limits.txt'
data = np.loadtxt(infile,unpack=True)
energy = data[0]

# Plot the limits for specific targets
# Draco = 9, Segue 1 = 19
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log',nonposy='clip')

targets = [ ('Draco', 9), ('Segue 1', 19) ]
for name,idx in targets:
    yerr = [0.3*data[idx], plt.zeros(len(data[idx]))]
    ax.errorbar(energy, data[idx], yerr=yerr, lolims=True, lw=1.25, ls='none', label=name)

plt.xlabel('Energy (MeV)')
plt.ylabel('Energy Flux $\mathrm{(MeV\,cm^{-2}\,s^{-1})}$')
plt.legend(loc='upper left')
outfile = 'example2.png'
print('Saving figure to %s...' % outfile)
plt.savefig(outfile)
plt.show()

