import numpy as np
import sncosmo as snc

class G10(snc.PropagationEffect):
    '''G10 smearing effect for sncosmo
       Use colordisp file of salt and follow SNANA formalism, see arXiv:1209.2482
    '''
    _param_names = ['L0', 'F0', 'F1', 'dL','RndS']
    param_names_latex = ['\lambda_0', 'F_0', 'F_1','d_L','RS']

    def __init__(self,model):
        self._parameters = np.array([2157.3,0.0,1.08e-4,800,np.random.randint(low=1000, high=100000)])
        self._minwave = model.source.minwave()
        self._maxwave = model.source.maxwave()
        self._colordisp = model.source._colordisp

    @property
    def scattering_law(self):
        '''Initialise scattering law'''
        L0,F0,F1,dL = self._parameters[:-1]
        lam = self._minwave
        sigma_lam=[]
        sigma_val=[]

        while lam < self._maxwave:
            sigma_lam.append(lam)
            val = self._colordisp(lam)
            if lam > L0:
                val*=1+(lam-L0)*F1
            elif lam < L0:
                val*=1+(lam-L0)*F0

            sigma_val.append(val)
            lam += self._parameters[3]
        return np.asarray(sigma_lam), np.asarray(sigma_val)

    @property
    def lam_scatter(self):
        '''Generate a smear funtion'''
        sigma_lam, sigma_val  = self.scattering_law
        RS = self._parameters[-1]
        scatter = sigma_val*np.random.default_rng(int(RS)).normal(0,1,size=len(sigma_val))
        return sigma_lam, sigma_val*np.random.default_rng(int(RS)).normal(0,1,size=len(sigma_val))

    def propagate(self, wave, flux):
        """Propagate the flux."""
        lam, scatter = self.lam_scatter
        smear = np.asarray([ut.sine_interp(w, lam, scatter) for w in wave])
        return flux*10**(-0.4*smear)


class C11(snc.PropagationEffect):
    '''C11 smearing effect for sncosmo
       Use COV matrix from N. Chottard thesis and follow SNANA formalism, see arXiv:1209.2482
    '''
    _param_names = ['Cuu','Sc','RndS']
    param_names_latex = ["\rho_{u'u}",'Sc','RS']

    def __init__(self,model):
        self._parameters = np.array([0., 1.3, np.random.randint(low=1000, high=100000)])
        self._minwave = model.source.minwave()
        self._maxwave = model.source.maxwave()

        # U'UBVRI lambda eff
        self.sigma_lam = np.array([2500.0, 3560.0, 4390.0, 5490.0, 6545.0, 8045.0])
        # U'UBVRI correlation matrix extract from SNANA, came from N.Chotard thesis
        self._CORR_matrix = np.array([[+1.000, 0.000,     0.000,     0.000,     0.000,     0.000],
                                     [0.000, +1.000000, -0.118516, -0.768635, -0.908202, -0.219447],
                                     [0.000, -0.118516, +1.000000, +0.570333, -0.238470, -0.888611],
                                     [0.000, -0.768635, +0.570333, +1.000000, +0.530320, -0.399538],
                                     [0.000, -0.908202, -0.238470, +0.530320, +1.000000, +0.490134],
                                     [0.000, -0.219447, -0.888611, -0.399538, +0.490134, +1.000000]])
        # U'UBVRI sigma
        self._sigma = np.array([ 0.5900, 0.06001, 0.040034, 0.050014, 0.040017, 0.080007 ])
        return

    @property
    def covmat(self):
        covmat = np.zeros((6,6))
        for i in range(6):
            for j in range(i+1):
                cor2cov = self._CORR_matrix[i,j]
                if  i != 0 and j == 0:
                    cor2cov = self._parameters[0] * self._CORR_matrix[i,1]
                sigi_sigj = self._sigma[i]*self._sigma[j]
                cor2cov *= sigi_sigj
                covmat[i,j] = covmat[j,i] = cor2cov * self._parameters[1]
        return covmat

    @property
    def scatter(self):
        '''Use the cov matrix to generate 6 random numbers'''
        RS = self._parameters[-1]
        mu = np.zeros(6)
        scat = np.random.default_rng(int(RS)).multivariate_normal(mu, self.covmat, check_valid='raise')
        return scat

    def propagate(self, wave, flux):
        if self._parameters[0] not in [0., 1., -1.]:
            raise ValueError('Cov_uu must be 1,-1 or 0')

        scatter = self.scatter
        smear = np.zeros(len(wave))
        for i,w in enumerate(wave):
            if w >= self.sigma_lam[-1]:
                smear[i] = catter[-1]
            elif w <= self.sigma_lam[0]:
                smear[i] = scatter[0]
            else:
                smear[i] = ut.sine_interp(w, self.sigma_lam, scatter)
        return flux*10**(-0.4*smear)
