"""Contains transients models."""
import copy
import numpy as np
from .constants import C_LIGHT_KMS


class BasicTransient:
    def __init__(self, parameters, sim_model, model_par):
        self.sim_model = copy.copy(sim_model)
        self._params = parameters
        self._model_par = model_par

        self._epochs = None
        self._sim_lc = None
        self._ID = None

    @property
    def ID(self):
        """Get SN ID."""
        return self._ID

    @ID.setter
    def ID(self, ID):
        """Set SN ID."""
        if isinstance(ID, int):
            self._ID = ID
        else:
            print('SN ID must be an integer')
        if self.sim_lc is not None:
            self.sim_lc.attrs['sn_id'] = self._ID

    @property
    def sim_t0(self):
        """Get SN peakmag time."""
        return self._params['sim_t0']

    @property
    def vpec(self):
        """Get SN peculiar velocity."""
        return self._params['vpec']

    @property
    def zcos(self):
        """Get SN cosmological redshift."""
        return self._params['zcos']

    @property
    def como_dist(self):
        """Get SN comoving distance."""
        return self._params['como_dist']

    @property
    def coord(self):
        """Get SN coordinates (ra,dec)."""
        return self._params['ra'], self._params['dec']

    @property
    def mag_sct(self):
        """Get SN coherent scattering term."""
        return self._params['mag_sct']

    @property
    def zpec(self):
        """Get SN peculiar velocity redshift."""
        return self.vpec / C_LIGHT_KMS

    @property
    def zCMB(self):
        """Get SN CMB frame redshift."""
        return (1 + self.zcos) * (1 + self.zpec) - 1.

    @property
    def z2cmb(self):
        """Get SN redshift due to our motion relative to CMB."""
        return self._params['z2cmb']

    @property
    def zobs(self):
        """Get SN observed redshift."""
        return (1 + self.zcos) * (1 + self.zpec) * (1 + self.z2cmb) - 1.

    @property
    def epochs(self):
        """Get SN observed redshift."""
        return self._epochs

    @epochs.setter
    def epochs(self, ep_dic):
        """Get SN observed epochs."""
        self._epochs = ep_dic

    @property
    def sim_mu(self):
        """Get SN distance moduli."""
        return 5 * np.log10((1 + self.zcos) * (1 + self.z2cmb) *
                            (1 + self.zpec)**2 * self.como_dist) + 25

    @property
    def sim_lc(self):
        """Get sim_lc."""
        return self._sim_lc


class SNIa(BasicTransient):
    _available_models = ['salt2', 'salt3']
