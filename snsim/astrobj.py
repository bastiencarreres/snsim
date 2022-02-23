"""Contains transients models."""

import copy
import abc
import numpy as np
import pandas as pd
from .constants import C_LIGHT_KMS
from . import nb_fun as nbf
from . import dust_utils as dst_ut


class BasicAstrObj(abc.ABC):
    """Short summary.

    Parameters
    ----------
    parameters : dict
        Parameters of the obj.

      | parameters
      | ├── zcos, cosmological redshift
      | ├── z2cmb, CMB dipole redshift contribution
      | ├── como_dist, comoving distance of the obj
      | ├── vpec, obj peculiar velocity
      | ├── ra, obj Right Ascension
      | ├── dec, obj Declinaison
      | ├── sim_t0, obj peak time
      | ├── dip_dM, magnitude modification due to a dipole
      | └── sncosmo
      |     └── sncosmo parameters
    sim_model : sncosmo.Model
        sncosmo Model to use.
    model_par : dict
        General model parameters.
        model_par
        └── mod_fcov, boolean to use or not model flux covariance
    """

    _type = ''

    def __init__(self, parameters, sim_model, model_par):
        # -- sncosmo model
        self.sim_model = copy.copy(sim_model)

        # -- Intrinsic parameters of the astrobj
        self._params = parameters
        self._model_par = model_par
        self._update_model_par()

        if 'dip_dM' in self._params:
            self.dip_dM = self._params['dip_dM']

        # -- set parameters of the sncosmo model
        self._set_model()

        # -- Define attributes to be set
        self._epochs = None
        self._sim_lc = None
        self._ID = None

    def _set_model(self):
        # Set sncosmo model parameters
        params = {**{'z': self.zobs, 't0': self.sim_t0},
                  **self._params['sncosmo']}
        self.sim_model.set(**params)

    @abc.abstractmethod
    def _update_model_par(self):
        """Abstract method to add general model parameters call during __init__."""
        pass

    def _add_meta_to_table(self):
        """Optional method to add metadata to sim_lc,
        called during _reformat_sim_table.
        """
        pass

    def gen_flux(self, rand_gen):
        """Generate the obj lightcurve.

        Parameters
        ----------
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        None

        Notes
        -----
        Set the sim_lc attributes as an astropy Table
        """
        random_seeds = rand_gen.integers(1000, 100000, size=2)

        # Re - set the parameters to take possible change (e.g dust)
        self._set_model()

        if self._model_par['mod_fcov']:
            # -- Implement the flux variation due to simulation model covariance
            gen = np.random.default_rng(random_seeds[0])
            fluxtrue, fluxcov = self.sim_model.bandfluxcov(self.epochs['band'],
                                                           self.epochs['time'],
                                                           zp=self.epochs['zp'],
                                                           zpsys=self.epochs['zpsys'])

            fluxtrue += gen.multivariate_normal(np.zeros(len(fluxcov)),
                                                fluxcov,
                                                check_valid='ignore',
                                                method='eigh')

        else:
            fluxtrue = self.sim_model.bandflux(self.epochs['band'],
                                               self.epochs['time'],
                                               zp=self.epochs['zp'],
                                               zpsys=self.epochs['zpsys'])

        # -- Noise computation : Poisson Noise + Skynoise + ZP noise
        fluxerr = np.sqrt(np.abs(fluxtrue) / self.epochs.gain
                          + self.epochs.skynoise**2
                          + (np.log(10) / 2.5 * fluxtrue * self.epochs.sig_zp)**2)

        gen = np.random.default_rng(random_seeds[1])
        flux = fluxtrue + gen.normal(loc=0., scale=fluxerr)

        # Set magnitude
        mag = np.zeros(len(flux))
        magerr = np.zeros(len(flux))

        positive_fmask = pd.eval('flux > 0')
        flux_pos = flux[positive_fmask]

        mag[positive_fmask] = -2.5 * np.log10(flux_pos) + self.epochs['zp'][positive_fmask]

        magerr[positive_fmask] = 2.5 / np.log(10) * 1 / flux_pos * fluxerr[positive_fmask]

        mag[~positive_fmask] = np.nan
        magerr[~positive_fmask] = np.nan

        # Create astropy Table lightcurve
        self._sim_lc = pd.DataFrame({'time': self.epochs['time'],
                                     'fluxtrue': fluxtrue,
                                     'flux': flux,
                                     'fluxerr': fluxerr,
                                     'mag': mag,
                                     'magerr': magerr,
                                     'zp': self.epochs['zp'],
                                     'zpsys': self.epochs['zpsys'],
                                     'gain': self.epochs['gain'],
                                     'skynoise': self.epochs['skynoise']})

        #self._sim_lc['mag'] = pd.eval('-2.5 * log10(self._sim_lc.flux) + self.epochs.zp')
        #self._sim_lc['magerr'] = pd.eval('2.5 / log(10) * 1 / self._sim_lc.flux * self._sim_lc.fluxerr')

        self._sim_lc.attrs = {**self.sim_lc.attrs,
                              **{'zobs': self.zobs, 't0': self.sim_t0},
                              **self._params['sncosmo']}

        self._sim_lc.reset_index(inplace=True)
        self._sim_lc.index.set_names('epochs', inplace=True)
        return self._reformat_sim_table()

    def _reformat_sim_table(self):
        """Give the good format to the sncosmo output Table.

        Returns
        -------
        None

        Notes
        -----
        Directly change the sim_lc attribute

        """
        # Keys that don't need renaming
        not_to_change = ['G10', 'C11', 'mw_']
        dont_touch = ['zobs', 'mw_r_v', 'fcov_seed']

        for k in self.epochs.columns:
            if k not in self.sim_lc.columns:
                self._sim_lc[k] = self.epochs[k].to_numpy()

        for k in self.sim_lc.attrs.copy():
            if k not in dont_touch and k[:3] not in not_to_change:
                self.sim_lc.attrs['sim_' + k] = self.sim_lc.attrs.pop(k)

        self.sim_lc.attrs['type'] = self._type

        if self.ID is not None:
            self.sim_lc.attrs['ID'] = self.ID

        self.sim_lc.attrs['vpec'] = self.vpec
        self.sim_lc.attrs['zcos'] = self.zcos
        self.sim_lc.attrs['zpec'] = self.zpec
        self.sim_lc.attrs['z2cmb'] = self.z2cmb
        self.sim_lc.attrs['zCMB'] = self.zCMB
        self.sim_lc.attrs['ra'] = self.coord[0]
        self.sim_lc.attrs['dec'] = self.coord[1]
        self.sim_lc.attrs['sim_mb'] = self.sim_mb
        self.sim_lc.attrs['sim_mu'] = self.sim_mu
        self.sim_lc.attrs['com_dist'] = self.como_dist

        if 'dip_dM' in self._params:
            self.sim_lc.attrs['dip_dM'] = self.dip_dM

        self._add_meta_to_table()

    @property
    def ID(self):
        """Get ID."""
        return self._ID

    @ID.setter
    def ID(self, ID):
        """Set ID."""
        if isinstance(ID, int):
            self._ID = ID
        else:
            print('SN ID must be an integer')
        if self.sim_lc is not None:
            self.sim_lc.attrs['ID'] = self._ID

    @property
    def sim_t0(self):
        """Get peakmag time."""
        return self._params['sim_t0']

    @property
    def vpec(self):
        """Get peculiar velocity."""
        return self._params['vpec']

    @property
    def zcos(self):
        """Get cosmological redshift."""
        return self._params['zcos']

    @property
    def como_dist(self):
        """Get comoving distance."""
        return self._params['como_dist']

    @property
    def coord(self):
        """Get coordinates (ra,dec)."""
        return self._params['ra'], self._params['dec']

    @property
    def mag_sct(self):
        """Get coherent scattering term."""
        return self._params['mag_sct']

    @property
    def zpec(self):
        """Get peculiar velocity redshift."""
        return self.vpec / C_LIGHT_KMS

    @property
    def zCMB(self):
        """Get CMB frame redshift."""
        return (1 + self.zcos) * (1 + self.zpec) - 1.

    @property
    def z2cmb(self):
        """Get redshift due to our motion relative to CMB."""
        return self._params['z2cmb']

    @property
    def zobs(self):
        """Get observed redshift."""
        return (1 + self.zcos) * (1 + self.zpec) * (1 + self.z2cmb) - 1.

    @property
    def epochs(self):
        """Get observed redshift."""
        return self._epochs

    @epochs.setter
    def epochs(self, ep_dic):
        """Get observed epochs."""
        self._epochs = ep_dic

    @property
    def sim_mu(self):
        """Get distance moduli."""
        return 5 * np.log10((1 + self.zcos) * (1 + self.z2cmb) *
                            (1 + self.zpec)**2 * self.como_dist) + 25

    @property
    def sim_lc(self):
        """Get sim_lc."""
        return self._sim_lc


class SNIa(BasicAstrObj):
    """SNIa object class.

    Parameters
    ----------
    sn_par : dict
        Parameters of the SN.

      | same as BasicAstrObj parameters
      | └── mag_sct, coherent mag scattering.
    sim_model : sncosmo.Model
        sncosmo Model to use.
    model_par : dict
        General model parameters.

      | same as BasicAstrObj model_par
      | ├── M0, SNIa absolute magnitude
      | ├── sigM, sigma of coherent scattering
      | └── used model parameters
    """
    _type = 'snIa'
    _available_models = ['salt2', 'salt3']

    def __init__(self, sn_par, sim_model, model_par):
        super().__init__(sn_par, sim_model, model_par)

    def _add_meta_to_table(self):
        """Add metadata to sim_lc."""
        self.sim_lc.attrs['m_sct'] = self.mag_sct

    def _update_model_par(self):
        """Extract and compute SN parameters that depends on used model.

        Notes
        -----
        Set attributes dependant on SN model
        SALT:
            - alpha -> _model_par['alpha']
            - beta -> _model_par['beta']
            - mb -> self.sim_mb
            - x0 -> self.sim_x0
            - x1 -> self.sim_x1
            - c -> self.sim_c
        """
        M0 = self._model_par['M0'] + self.mag_sct
        if self.sim_model.source.name in ['salt2', 'salt3']:
            # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
            # beta*c : scattering due to color and stretch} + {coherent intrinsic scattering}
            alpha = self._model_par['alpha']
            beta = self._model_par['beta']
            x1 = self._params['sncosmo']['x1']
            c = self._params['sncosmo']['c']
            mb = self.sim_mu + M0 - alpha * x1 + beta * c

            self.sim_x1 = x1
            self.sim_c = c

            if 'dip_dM' in self._params:
                mb += self._params['dip_dM']

            self.sim_mb = mb

            # Compute the x0 parameter
            self.sim_model.set(x1=self.sim_x1, c=self.sim_c)
            self.sim_model.set_source_peakmag(self.sim_mb, 'bessellb', 'ab')
            self.sim_x0 = self.sim_model.get('x0')
            self._params['sncosmo']['x0'] = self.sim_x0


    @property
    def mag_sct(self):
        """Get SN coherent scattering term."""
        return self._params['mag_sct']
