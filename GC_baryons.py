import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def baryon_profile(r, z):
    # r is a 2-d array of the r-values used in the grid of points
    # z is a 2-d array of the z-values used in the grid of points     
    zthin = 0.3 #in kpc
    zthick = 0.9 #in kpc
    rthin = 2.9 #in kpc
    rthick = 3.3 #in kpc
    rcut = 2.1 #in kpc
    r0 = 0.075 #in kpc
    rhobulge = 9.9e10 #in M_sun / kpc^3
    sthin = 8.17e8 #in M_sun / kpc^2
    sthick = 2.09e9 #in M_sun / kpc^2    
    q = 0.5     
    a = 1.8
    rho_baryon = rhobulge*np.exp(-(r**2 + z**2/q**2)/rcut**2) / (1 + np.sqrt(r**2 + z**2/q**2)/r0)**a + 0.5*sthin*np.exp(-z/zthin-r/rthin)/zthin + 0.5*sthick*np.exp(-z/zthick-r/rthick)/zthick
    return rho_baryon

# calculating \int dV rho(r,z) = 4\pi \int_0^rmax dz \int_0^rmax dr r rho(r,z)
rmax = np.sin(3.5*np.pi/180.)*8.25

print baryon_profile(0,0)
print baryon_profile(0.1,0.1)

n_r = 800
n_z = 800

r = np.linspace(0,rmax,n_r,endpoint=True)
z = np.linspace(0,rmax,n_z,endpoint=True)

integrand = np.zeros((n_r,n_z))
dens = np.zeros((n_r,n_z))
for i in range(n_r):
    for j in range(n_z):
        dens[i,j] = baryon_profile(r[i],z[j])
        integrand[i,j] = r[i]*baryon_profile(r[i],z[i])

mass = 4*np.pi*np.trapz(np.trapz(integrand, x=r, axis=0), x=z, axis=0)

print 'the mass with the cylindrical volume of '+str(rmax)+' kpc is '+str(mass)+' solar masses'

fig = plt.figure()
ax = fig.gca(projection='3d')

# Mesh data.
r, z = np.meshgrid(r, z)

# Plot the surface.
surf = ax.plot_surface(r, z, dens, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('baryon_density.png')
plt.clf()
