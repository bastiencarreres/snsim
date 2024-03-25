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
        cosmology = {"name": "planck18"}
        cosmo = snsim.utils.set_cosmo(cosmology)

        # Fake position
        zcos = 0.1
        coords = [0.0, 0.0]

        # Params dic
        sim_par = {
            "zcos": zcos,
            "zpcmb": 0.0,
            "como_dist": cosmo.comoving_distance(zcos).value,
            "vpec": 300,
            "t0": 0,  # simulated peak time of the event
            "ra": coords[0],
            "dec": coords[1],
            "coh_sct": 0.0,
            "x1": 1,
            "c": 0.1,
            "M0": -19.3,
            "alpha": 0.14,
            "beta": 3.1,
            "model_name": "salt2",
            "model_version": "2.4",
        }

        self.SNIa_Tripp = sn_astrobj.SNIa(sim_par, relation="SALTTripp")

        self.obs = pd.DataFrame(
            {
                "time": [-10, 0, 20, 50],
                "band": ["bessellb", "bessellv", "bessellr", "besselli"],
                "zp": np.ones(4) * 30,
                "zpsys": ["ab"] * 4,
                "gain": np.ones(4),
                "skynoise": np.zeros(4),
                "sig_zp": np.zeros(4),
            }
        )

    def test_tripp(self):
        mb = self.SNIa_Tripp.sim_par["M0"] + self.SNIa_Tripp.mu
        mb -= self.SNIa_Tripp.sim_par["alpha"] * self.SNIa_Tripp.sim_par["x1"]
        mb += self.SNIa_Tripp.sim_par["beta"] * self.SNIa_Tripp.sim_par["c"]
        assert self.SNIa_Tripp.mb == mb

    def test_genflux(self):
        lcs = self.SNIa_Tripp.gen_flux(self.obs, seed=1234, mod_fcov=False)
        print(lcs)
        test = {
            "fluxtrue": np.array([15425.39490416, 28576.9759029, 15492.50307286, 4990.21114817]),
            "fluxerrtrue": np.array([124.1990133, 169.04725938, 124.46888395, 70.64142657]),
            "flux": np.array([15559.40785314, 28473.85136389, 15251.03732738, 4972.76245161]),
            "fluxerr": np.array([124.73735548, 168.74196681, 123.4950903, 70.51781655]),
        }

        for k in ["flux", "fluxerr", "fluxtrue", "fluxerrtrue"]:
            assert_allclose(lcs[k].values, test[k])
