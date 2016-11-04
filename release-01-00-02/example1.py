#!/usr/bin/env python
#
# example1.py
#
# This example demonstrates how to load 
# and plot the dark matter limit files.
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

# Load channel-dependent limits on the cross section
infile = 'limits_bb.txt'
data   = np.loadtxt(infile,unpack=True)
mass   = data[0]

# Plot the limits for specific targets
# Draco = 9, Segue 1 = 19, Combined = 26
targets = [ ('Draco', 9), ('Segue 1', 19), ('Combined', 26) ]
for name,idx in targets:
    plt.loglog(mass, data[idx], '-', lw=2, label=name)

plt.xlabel('Mass (GeV)')
plt.ylabel(r'$\mathrm{\langle \sigma v \rangle\ (cm^3\,s^{-1})}$')
plt.legend(loc='upper left')
outfile = 'example1.png'
print('Saving figure to %s...' % outfile)
plt.savefig(outfile)
plt.show()

