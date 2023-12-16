import snsim
import sncosmo as snc
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_approx_equal
from snsim import astrobj as sn_astrobj

class TestSNIa:
    def setup_class(self):
        """Create a SNIa."""
        
        # Set the cosmology (astropy.cosmology object)
        cosmology = {'name': 'planck18'}
        cosmo =  snsim.utils.set_cosmo(cosmology)
        
        # Fake position
        zcos = 0.1
        coords = [0., 0.]

        # Params dic
        sim_par = {
            'zcos': zcos,
            'z2cmb': 0.0,
            'como_dist': cosmo.comoving_distance(zcos).value,
            'vpec': 300,
            't0': 0,#simulated peak time of the event
            'ra': coords[0],
            'dec': coords[1],
            'mag_sct': 0.0,
            'x1':1, 
            'c':0.1,
            'M0': -19.3,
            'alpha': 0.14,
            'beta': 3.1,
            'mod_fcov': False          
            }

        source = snc.get_source('salt2')
        
        self.SNIa_Tripp = sn_astrobj.SNIa(source, {'relation': 'tripp', **sim_par})
        
        self.obs = pd.DataFrame({
                    'time': [-10, 0, 20, 50],
                    'band': ['bessellb', 'bessellv', 'bessellr', 'besselli'],
                    'zp': np.ones(4) * 30,
                    'zpsys': ['ab'] * 4,
                    'gain': np.ones(4),
                    'skynoise': np.zeros(4),
                    'sig_zp': np.zeros(4)
                    })
    
    def test_tripp(self):
        mb = self.SNIa_Tripp.sim_par['M0'] + self.SNIa_Tripp.mu
        mb += self.SNIa_Tripp.sim_par['alpha'] * self.SNIa_Tripp.sim_par['x1']
        mb += -self.SNIa_Tripp.sim_par['beta'] * self.SNIa_Tripp.sim_par['c']
        assert(self.SNIa_Tripp.mb == mb)
    
    def test_genflux(self):
        lcs = self.SNIa_Tripp.gen_flux(self.obs, seed=1234)
        
        test = {
                'flux': np.array([15321.44241962, 28319.67362936, 15241.38760919,  4796.31358256]),
                'fluxerr': np.array([123.77981427, 168.28450205, 123.45601488,  69.25542277]),
                'fluxtrue': np.array([15188.46266493, 28422.51909986, 15482.77755192,  4813.45046363]),
                'fluxerrtrue': np.array([123.2414811 , 168.58979536, 124.42980974,  69.37903476])
                }
        
        for k in ['flux', 'fluxerr', 'fluxtrue', 'fluxerrtrue']:
            assert_allclose(lcs[k].values, test[k])
            
    

        
    