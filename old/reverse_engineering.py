import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt


like_name = np.array(['like_bootes_I',
                        'like_bootes_II',
                        'like_bootes_III',
                        'like_canes_venatici_I',
                        'like_canes_venatici_II',
                        'like_canis_major',
                        'like_carina',
                        'like_coma_berenices',
                        'like_draco',
                        'like_fornax',
                        'like_hercules',
                        'like_leo_I',
                        'like_leo_II',
                        'like_leo_IV',
                        'like_leo_V',
                        'like_pisces_II',
                        'like_sagittarius',
                        'like_sculptor',
                        'like_segue_1',
                        'like_segue_2',
                        'like_sextans',
                        'like_ursa_major_I',
                        'like_ursa_major_II',
                        'like_ursa_minor',
                        'like_willman_1'])





data = np.loadtxt('release-01-00-02/like_draco.txt',unpack=True)

emin = data[0]
emax = data[1]
eflux = data[2]
delta_loglike = data[3]



print emin.shape

eavg = 0.5*(emax-emin)

nflux = eflux / eavg


print 'the number flux is...'
print nflux[:25]

def test_log_like(k, b, e, Labs, f):
    return k*np.log((f + b)*e) - (f + b)*e - k*np.log(k) + k - Labs

print 'the test of the log-like at the initial guess is...'
print test_log_like(1000,2.e-10,2e13,100.,nflux[:25])

print 'the data vector of the log-like is...'
print delta_loglike[:25]

fit_values = np.zeros((24,4))

for i in range(24):
    istart = i*25
    iend = istart+25
    def func(x):
        dloglike = delta_loglike[istart:iend]
        f = nflux[istart:iend]
        return np.sum((test_log_like(x[0],x[1],x[2],x[3],f) - dloglike)**2)

    x0 = np.array([1000, 2.e-10, 2.e13, 100.])
    bnds = ((0,None), (0,None), (0,None), (None,None))
    res = scop.minimize(func,x0,method = 'Nelder-Mead')
    #print func(res.x)
    print res.x
    fit_values[i,:] = res.x

np.savetxt('draco_data.txt',fit_values)

print test_log_like(fit_values[0,0],fit_values[0,1],fit_values[0,2],fit_values[0,3],nflux[:25])

print delta_loglike[:25]

plt.plot(nflux[:25],delta_loglike[:25],label = 'Fermi data')
plt.plot(nflux[:25],test_log_like(fit_values[0,0],fit_values[0,1],fit_values[0,2],fit_values[0,3],nflux[:25]),label = 'Reconstruction')
plt.xlabel(r'Number flux [cm$^{-2}$ sec$^{-1}$]')
plt.ylabel(r'$\Delta$log-likelihood')
plt.savefig('test_reconstruction.png')



for name in like_name:
    print name
    data = np.loadtxt('release-01-00-02/'+name+'.txt',unpack=True)
    emin = data[0]
    emax = data[1]
    assert len(emin) == 24*25
    eflux = data[2]
    delta_loglike = data[3]
    eavg = 0.5*(emax-emin)
    nflux = eflux / eavg
    fit_values = np.zeros((24,4))
    for i in range(24):
        istart = i*25
        iend = istart+25
        def func(x):
            dloglike = delta_loglike[istart:iend]
            f = nflux[istart:iend]
            return np.sum((test_log_like(x[0],x[1],x[2],x[3],f) - dloglike)**2)
        x0 = np.array([1000, 2.e-10, 2.e13, 100.])
        bnds = ((0,None), (0,None), (0,None), (None,None))
        res = scop.minimize(func,x0,method = 'Nelder-Mead')
        fit_values[i,:] = res.x
        assert fit_values[i,0] > 0, 'number count is not positive'
        assert fit_values[i,1] > 0, 'exposure is not positive'
        assert fit_values[i,2] > 0, ' background flux is not positive'
    np.savetxt('dwarf_re_data/'+name+'_data.txt',fit_values)


#def test_log_like_gauss(k, b, e, Labs, f):
#    return -0.5*(k - (b+f)*e)**2/k - Labs

#fit_values_gauss = np.zeros((24,4))
#print 'now trying with a gaussian likelihood'

#for i in range(24):
#    istart = i*25
#    iend = istart+25
#    def func(x):
#        dloglike = delta_loglike[istart:iend]
#        f = nflux[istart:iend]
#        return np.sum((test_log_like_gauss(x[0],x[1],x[2],x[3],f) - dloglike)**2)
#    x0 = np.array([1000, 2.e-10, 2.e13, 100.])
#    bnds = ((0,None), (0,None), (0,None), (None,None))
#    res = scop.minimize(func,x0,method = 'Nelder-Mead')
    #print func(res.x)
    #print res.x
#    fit_values_gauss[i,:] = res.x

#print test_log_like(res.x[0],res.x[1],res.x[2],res.x[3],nflux[:25])

#print delta_loglike[:25]

#plt.plot(nflux[:25],delta_loglike[:25])
#plt.plot(nflux[:25],test_log_like(res.x[0],res.x[1],res.x[2],res.x[3],nflux[:25]))
#plt.savefig('test_reconstruction.png')
