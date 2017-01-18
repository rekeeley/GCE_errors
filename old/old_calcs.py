def density(x,y,scale_radius,  gamma): #NFW density profile
    solar_radius = 8.25 #x is the los distance / solar radius  as to make the integral dimensionless
    R = np.sqrt(1.-2.*y*x + x*x)#R / solar radius
    return pow(R,-gamma)*pow((1 + R*solar_radius/scale_radius)/(1+solar_radius/scale_radius),gamma-3)

def J_factor(scale_radius,local_density,gamma):
    #integrating density^2 from x=0 to inf; x is line of sight distance/solarradius
    #theta = np.zeros((7,7))
    temp = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            y = np.cos(.5*(1+i+j)*np.pi/180.)
            integrand = lambda x: density(x,y,scale_radius,gamma)**2
            ans, err = integrate.quad(integrand, 0,np.inf)
            temp[i,j] = ans
    J = temp.mean()
    kpctocm = 3.08568e21
    deltaomega = (7.*np.pi/180.)**2
    return  deltaomega*J*8.25*kpctocm*local_density*local_density
    
def conc():
    coeff = np.array([37.5153,-1.5093,1.63e-2,3.66e-4,-2.89237e-5,5.32e-7])
    h = 0.67
    Mmw = 1.5e12
    rvir = 200.
    conc = 0.
    for i in range(6):
        conc += coeff[i]*np.log(h*Mmw)**i
    return conc    
    
    def get_dn_de_log_parab(N0,alpha,beta,Eb,energy):
    n_spec = len(energy)
    n_eb = len(Eb)
    n_beta = len(beta)
    n_alpha = len(alpha)
    n_N0 = len(N0)
    N0 = np.tile(N0[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],(1,n_alpha,n_beta,n_eb,n_spec))
    alpha = np.tile(alpha[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis],(n_N0,1,n_beta,n_eb,n_spec))
    beta = np.tile(beta[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis],(n_N0,n_alpha,1,n_eb,n_spec))
    Eb = np.tile(Eb[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis],(n_N0,n_alpha,n_beta,1,n_spec))
    energy = np.tile(energy[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:],(n_N0,n_alpha,n_beta,n_eb,1))
    return N0*(energy/Eb)**(-alpha - beta*np.log(energy/Eb))
