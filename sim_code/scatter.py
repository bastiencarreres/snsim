import numpy as np
import sncosmo as snc
from sim_utils import sine_interp

class C11(snc.PropagationEffect):
    '''C11 smearing effect for sncosmo
       Use COV matrix from N. Chottard thesis and follow SNANA formalism, see arXiv:1209.2482
    '''
    _param_names = ['Cuu','Sc','RndS']
    param_names_latex = ["\rho_{u'u}",'Sc','RS']

    def __init__(self,model):
        self._parameters = np.array([0.,1.3,np.random.randint(low=1000, high=100000)])
        self.tmp_par = self._parameters[:-1].copy()
        self.check_coruu()
        self._minwave = model.source.minwave()
        self._maxwave = model.source.maxwave()

        # U'UBVRI lambda eff
        self.sigma_lam = np.array([2500.0, 3560.0, 4390.0, 5490.0, 6545.0, 8045.0])
        # U'UBVRI correlation matrix extract from SNANA, came from N.Chotard thesis
        self.CORR_matrix = np.array([[+1.000, 0.000,     0.000,     0.000,     0.000,     0.000],
                                     [0.000, +1.000000, -0.118516, -0.768635, -0.908202, -0.219447],
                                     [0.000, -0.118516, +1.000000, +0.570333, -0.238470, -0.888611],
                                     [0.000, -0.768635, +0.570333, +1.000000, +0.530320, -0.399538],
                                     [0.000, -0.908202, -0.238470, +0.530320, +1.000000, +0.490134],
                                     [0.000, -0.219447, -0.888611, -0.399538, +0.490134, +1.000000]])
        # U'UBVRI sigma
        self.sigma = np.array([ 0.5900, 0.06001, 0.040034, 0.050014, 0.040017, 0.080007 ])
        #Initialisation of covmatrix
        self.init_covmat()
        return

    def init_covmat(self):
        self.covmat = np.zeros((6,6))
        for i in range(6):
            for j in range(i+1):
                cor2cov = self.CORR_matrix[i,j]
                if  i != 0 and j == 0:
                    cor2cov = self._parameters[0] * self.CORR_matrix[i,1]
                sigi_sigj = self.sigma[i]*self.sigma[j]
                cor2cov *= sigi_sigj
                self.covmat[i,j] = self.covmat[j,i] = cor2cov*self._parameters[1]
        return

    def check_coruu(self):
        if self._parameters[0] not in [1.,0.,-1]:
            raise ValueError('C11_i can be 1, 0 or -1')
        comp_tmp = (self.tmp_par != self._parameters[:-1]).any()
        if comp_tmp:
            self.init_covmat()
            self.tmp_par = self._parameters[:-1].copy()
        return

    def gen_smearing(self):
        '''Use the cov matrix to generate 6 random numbers'''
        self.check_coruu()
        RS=self._parameters[-1]
        mu = np.zeros(6)
        self.sigma_scatter = np.random.default_rng(int(RS)).multivariate_normal(mu, self.covmat, check_valid='raise')
        return

    def propagate(self, wave, flux):
        self.gen_smearing()
        smear= np.zeros(len(wave))
        for i,w in enumerate(wave):
            if w >= self.sigma_lam[-1]:
                smear[i] = self.sigma_scatter[-1]
            elif w <= self.sigma_lam[0]:
                smear[i] = self.sigma_scatter[0]
            else:
                smear[i] = sine_interp(w, self.sigma_lam, self.sigma_scatter)
        return flux*10**(-0.4*smear)

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
        self.tmp_par = self._parameters[:-1].copy()
        self.scattering_law()

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

        self.sigma_lam = np.asarray(sigma_lam)
        self.sigma_val = np.asarray(sigma_val)
        return

    def gen_smearing(self):
        '''Generate a smear funtion'''
        comp_tmp = (self.tmp_par != self._parameters[:-1]).any()
        if comp_tmp:
            self.scattering_law()
            self.tmp_par = self._parameters[:-1].copy()
        RS=self._parameters[-1]
        self.sigma_scatter = self.sigma_val*np.random.default_rng(int(RS)).normal(0,1,size=len(self.sigma_val))
        return

    def propagate(self, wave, flux):
        """Propagate the flux."""
        self.gen_smearing()
        smear = np.asarray([sine_interp(w, self.sigma_lam, self.sigma_scatter) for w in wave])
        return flux*10**(-0.4*smear)

def add_filter(path):  # Not implemented yet for later purpose
    input_name = {}
    for band in bands:
        table = np.loadtxt(band[1])
        name = band[0]
        band = snc.Bandpass(wavelength, transmission, name=name)
        try:
            snc.register(band)
        except (Exception):
            band.name += '_temp'
            snc.register(band, force=True)
            input_name[band[0]] = band.name
    if input_name == {}:
        return None
    else:
        return input_name
