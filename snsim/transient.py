"""Contains transients models."""

import copy
import numpy as np
import pandas as pd
from .constants import C_LIGHT_KMS


class BasicTransient:
    def __init__(self, parameters, sim_model, model_par):
        self.sim_model = copy.copy(sim_model)
        self._params = parameters
        self._model_par = model_par

        if 'mw_' in self.sim_model.effect_names:
            self.mw_ebv = self._model_par['sncosmo']['mw_ebv']

        self._epochs = None
        self._sim_lc = None
        self._ID = None

    def _set_model(self):
        # Set sncosmo model parameters
        params = {**{'z': self.zobs, 't0': self.sim_t0},
                  **self._model_par['sncosmo']}
        self.sim_model.set(**params)

    def pass_cut(self, nep_cut):
        """Check if the Transient pass the given cuts.

        Parameters
        ----------
        nep_cut : list
            nep_cut = [[nep_min1,Tmin,Tmax],[nep_min2,Tmin2,Tmax2,'filter1'],...]

        Returns
        -------
        boolean
            True or False.

        """
        if self.epochs is None:
            return False
        else:
            for cut in nep_cut:
                cutMin_obsfrm, cutMax_obsfrm = cut[1] * (1 + self.zobs), cut[2] * (1 + self.zobs)
                test = (self.epochs['time'] - self.sim_t0 > cutMin_obsfrm)
                test *= (self.epochs['time'] - self.sim_t0 < cutMax_obsfrm)
                if len(cut) == 4:
                    test *= (self.epochs['band'] == cut[3])
                if np.sum(test) < int(cut[0]):
                    return False
            return True

    def gen_flux(self, rand_gen):
        """Generate the SN lightcurve.

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
        print(self.sim_model.parameters)

        if self._model_par['mod_fcov']:
            # Implement the flux variation due to simulation model covariance
            gen = np.random.default_rng(random_seeds[0])
            flux, fluxcov = self.sim_model.bandfluxcov(self.epochs['band'],
                                                       self.epochs['time'],
                                                       zp=self.epochs['zp'],
                                                       zpsys=self.epochs['zpsys'])

            flux += gen.multivariate_normal(np.zeros(len(fluxcov)),
                                            fluxcov,
                                            check_valid='ignore',
                                            method='eigh')

        else:
            flux = self.sim_model.bandflux(self.epochs['band'],
                                           self.epochs['time'],
                                           zp=self.epochs['zp'],
                                           zpsys=self.epochs['zpsys'])


        # Noise computation : Poisson Noise + Skynoise + ZP noise
        fluxerr = np.sqrt(np.abs(flux) / self.epochs['gain']
                          + self.epochs['skynoise']**2
                          + (np.log(10) / 2.5 * flux * self.epochs['sig_zp'])**2)

        gen = np.random.default_rng(random_seeds[1])
        flux += gen.normal(loc=0., scale=fluxerr)

        # Set magnitude
        mag = np.zeros(len(flux))
        magerr = np.zeros(len(flux))

        positive_flux = flux > 0

        mag[positive_flux] = -2.5 * np.log10(flux[positive_flux]) + self.epochs['zp'][positive_flux]

        magerr[positive_flux] = 2.5 / np.log(10) * 1 / flux[positive_flux] * fluxerr[positive_flux]

        mag[~positive_flux] = np.nan
        magerr[~positive_flux] = np.nan

        # Create astropy Table lightcurve
        self._sim_lc = pd.DataFrame({'time': self.epochs['time'],
                                     'flux': flux,
                                     'fluxerr': fluxerr,
                                     'mag': mag,
                                     'magerr': magerr,
                                     'zp': self.epochs['zp'],
                                     'zpsys': self.epochs['zpsys'],
                                     'gain': self.epochs['gain'],
                                     'skynoise': self.epochs['skynoise'],
                                     'epochs': np.arange(len(self.epochs['time']))
                                     })

        self._sim_lc.attrs = {**self.sim_lc.attrs,
                              **{'zobs': self.zobs, 't0': self.sim_t0},
                              **self._model_par['sncosmo']}

        self._sim_lc.set_index('epochs', inplace=True)
        return self._reformat_sim_table()

    def _add_meta_to_table():
        pass

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

        for k in self.epochs.keys():
            if k not in self.sim_lc.copy().keys():
                self._sim_lc[k] = self.epochs[k].copy()

        for k in self.sim_lc.attrs.copy():
            if k not in dont_touch and k[:3] not in not_to_change:
                self.sim_lc.attrs['sim_' + k] = self.sim_lc.attrs.pop(k)

        if self.ID is not None:
            self.sim_lc.attrs['id'] = self.ID

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
        self._add_meta_to_table()

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

    def __init__(self, sn_par, sim_model, model_par):
        super().__init__(sn_par, sim_model, model_par)
        self._init_model_par()
        super()._set_model()

    def _add_meta_to_table(self):
        self.sim_lc.attrs['m_sct'] = self.mag_sct

        if 'adip_dM' in self._params:
            self.sim_lc.attrs['adip_dM'] = self.adip_dM

    def _init_model_par(self):
        """Extract and compute SN parameters that depends on used model.

        Returns
        -------
        None

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
            x1 = self._model_par['sncosmo']['x1']
            c = self._model_par['sncosmo']['c']
            mb = self.sim_mu + M0 - alpha * x1 + beta * c

            # Compute the x0 parameter
            self.sim_model.set(x1=x1, c=c)
            self.sim_model.set_source_peakmag(mb, 'bessellb', 'ab')
            self.sim_x0 = self.sim_model.get('x0')
            self._model_par['sncosmo']['x0'] = self.sim_x0

            self.sim_x1 = x1
            self.sim_c = c

        # elif self.sim_model.source.name == 'snoopy':
            # TODO
        # Dipole
        if 'dip_dM' in self._sn_par:
            mb += self._sn_par['dip_dM']
            self.adip_dM = self._sn_par['dip_dM']

        self.sim_mb = mb

    @property
    def mag_sct(self):
        """Get SN coherent scattering term."""
        return self._params['mag_sct']
