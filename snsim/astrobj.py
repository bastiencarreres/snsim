"""Contains transients models."""

import copy
import abc
import numpy as np
import pandas as pd
from .constants import C_LIGHT_KMS
from . import utils as ut


class BasicAstrObj(abc.ABC):
    """Basic class for transients.

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
    _base_attrs = ['ID', 'coord', 'zcos', 'zCMB', 'zpec',
                   'vpec', 'z2cmb', 'sim_mu', 'como_dist']

    def __init__(self, parameters, sim_model, model_par):

        # -- Update attrs
        self._base_attrs = self._base_attrs + self._attrs

        # -- sncosmo model
        self.sim_model = copy.deepcopy(sim_model)

        # -- Intrinsic parameters of the astrobj
        self._params = parameters
        self._model_par = model_par
        self._update_model_par()

        if 'dip_dM' in self._params:
            self.dip_dM = self._params['dip_dM']

        # -- set parameters of the sncosmo model
        self._set_model()
        
    def _set_model(self):
        # Set sncosmo model parameters
        params = {**{'z': self.zobs, 't0': self.sim_t0},
                  **self._params['sncosmo']}
        self.sim_model.set(**params)

    @abc.abstractmethod
    def _update_model_par(self):
        """Abstract method to add general model parameters call during __init__."""
        pass

    def gen_flux(self, obs, seed=None):
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
        Set the sim_lc attributes as a pandas.DataFrame
        """
        if seed is None:
            random_seeds = np.random.randint(1e3, 1e6, size=2)
        else:
            random_seeds = np.random.default_rng(seed).integers(1e3, 1e6, size=2)

        # Re - set the parameters
        self._set_model()

        # mask to delete observed points outside time range of the model
        obs = obs[(obs['time']  > self.sim_model.mintime()) & (obs['time'] < self.sim_model.maxtime())]

        if self._model_par['mod_fcov']:
            # -- Implement the flux variation due to simulation model covariance
            gen = np.random.default_rng(random_seeds[0])
            fluxtrue, fluxcov = self.sim_model.bandfluxcov(obs['band'],
                                                           obs['time'],
                                                           zp=obs['zp'],
                                                           zpsys=obs['zpsys'])

            fluxtrue += gen.multivariate_normal(np.zeros(len(fluxcov)),
                                                fluxcov,
                                                check_valid='ignore',
                                                method='eigh')

        else:
            fluxtrue = self.sim_model.bandflux(obs['band'],
                                               obs['time'],
                                               zp=obs['zp'],
                                               zpsys=obs['zpsys'])

        # -- Noise computation : Poisson Noise + Skynoise + ZP noise
        fluxerrtrue = np.sqrt(np.abs(fluxtrue) / obs.gain
                          + obs.skynoise**2
                          + (np.log(10) / 2.5 * fluxtrue * obs.sig_zp)**2)

        gen = np.random.default_rng(random_seeds[1])
        flux = fluxtrue + gen.normal(loc=0., scale=fluxerrtrue)
        fluxerr = np.sqrt(np.abs(flux) / obs.gain
                          + obs.skynoise**2
                          + (np.log(10) / 2.5 * flux * obs.sig_zp)**2)

        # Set magnitude
        mag = np.zeros_like(flux)
        magerr = np.zeros_like(flux)

        positive_fmask = flux > 0
        flux_pos = flux[positive_fmask]

        mag[positive_fmask] = -2.5 * np.log10(flux_pos) + obs['zp'][positive_fmask]

        magerr[positive_fmask] = 2.5 / np.log(10) * 1 / flux_pos * fluxerr[positive_fmask]

        mag[~positive_fmask] = np.nan
        magerr[~positive_fmask] = np.nan

        # Create astropy Table lightcurve
        sim_lc = pd.DataFrame({'time': obs['time'],
                               'fluxtrue': fluxtrue,
                               'fluxerrtrue': fluxerrtrue,
                               'flux': flux,
                               'fluxerr': fluxerr,
                               'mag': mag,
                               'magerr': magerr,
                               'zp': obs['zp'],
                               'zpsys': obs['zpsys'],
                               'gain': obs['gain'],
                               'skynoise': obs['skynoise']})

        for k in obs.columns:
            if k not in sim_lc.columns:
                sim_lc[k] = obs[k].values

        sim_lc.attrs = {**sim_lc.attrs,
                        **{'zobs': self.zobs, 't0': self.sim_t0},
                        **self._params['sncosmo']}

        sim_lc.reset_index(inplace=True)
        sim_lc.index.set_names('epochs', inplace=True)
        return self._reformat_sim_table(sim_lc)

    def _reformat_sim_table(self, sim_lc):
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

        for k in sim_lc.attrs.copy():
            if k not in dont_touch and k[:3] not in not_to_change:
                sim_lc.attrs['sim_' + k] = sim_lc.attrs.pop(k)

        sim_lc.attrs['type'] = self._type

        for meta in self._base_attrs:
            if meta == 'coord':
                sim_lc.attrs['ra'] = self.coord[0]
                sim_lc.attrs['dec'] = self.coord[1]
            else:
                attrs = getattr(self, meta)
                if attrs is not None:
                    sim_lc.attrs[meta] = getattr(self, meta)

        if 'dip_dM' in self._params:
            sim_lc.attrs['dip_dM'] = self.dip_dM
        if 'template' in self._params:
            sim_lc.attrs['template']= self._params['template']

        return sim_lc

    @property
    def ID(self):
        """Get ID."""
        if 'ID' in self._params:
            return self._params['ID']

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
    def sim_mu(self):
        """Get distance moduli."""
        return 5 * np.log10((1 + self.zcos) * (1 + self.z2cmb) *
                            (1 + self.zpec)**2 * self.como_dist) + 25


class SNIa(BasicAstrObj):
    """SNIa class.

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
    _attrs = ['sim_mb', 'mag_sct']

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
            self._params['template']=self.sim_model.source.name
            # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
            # beta*c : scattering due to color and stretch} + {coherent intrinsic scattering}
            alpha = self._model_par['alpha']
            x1 = self._params['sncosmo']['x1']
            c = self._params['sncosmo']['c']

            if isinstance(self._model_par['beta'] , str):
                if self._model_par['beta'].lower() == 'bs20':
                    #in this case the beta parameter is included in the mag_sct
                    mb = self.sim_mu + M0 - alpha * x1
            else:
                #in this case beta is just 1 value for all SN
                beta = self._model_par['beta']
                mb = self.sim_mu + M0 - alpha * x1 + beta * c #add mass step if you have host 

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
        """SN coherent scattering term."""
        return self._params['mag_sct']


class TimeSeries(BasicAstrObj):
    """TimeSeries class.

    Parameters
    ----------
    sn_par : dict
        Parameters of the object.

      | same as BasicAstrObj parameters
      | └── mag_sct, coherent mag scattering.
    sim_model : sncosmo.Model
        sncosmo Model to use.
    model_par : dict
        General model parameters.

      | same as BasicAstrObj model_par
      | ├── M0,  absolute magnitude
      | ├── sigM, sigma of coherent scattering
      | └── used model parameters
    """
    _attrs = ['sim_mb', 'mag_sct']

    def _update_model_par(self):
        """Extract and compute SN parameters that depends on used model.

        Notes
        -----
        Set attributes dependant on SN model
       
            - mb -> self.sim_mb
            - amplitude -> self.sim_amplitude
            - Template -> self._params['template']  Sed template used 
           
        """
        M0 = self._model_par['M0'] +  self.mag_sct 
        self._params['M0'] = M0
        if self.sim_model.source.name in self._available_models:
            self._params['template']=self.sim_model.source.name
            m_r= self.sim_mu + M0
    
            if 'dip_dM' in self._params:
                m_r += self._params['dip_dM']

            # Compute the amplitude  parameter
            self.sim_model.set_source_peakmag(m_r, 'bessellr', 'ab')
            self.sim_mb = self.sim_model.source_peakmag( 'bessellb', 'ab')
            self.sim_amplitude = self.sim_model.get('amplitude')
            self._params['sncosmo']['amplitude'] = self.sim_amplitude


    @property
    def mag_sct(self):
        """SN coherent scattering term."""
        return self._params['mag_sct']
        
    @property
    def M0(self):
        """SN absolute magnitude in B-band"""
        return self._params['M0']

    @property
    def template(self):
        """sncosmo.Model source name """
        return self._params['template']
  

class SNII(TimeSeries):
    """SNII class.

    Parameters
    ----------
    same as TimeSeries class"""
    _type = 'snII'
    _available_models = ut.Templatelist_fromsncosmo('snii')

class SNIIpl(TimeSeries):
    """SNII P/L class.

    Parameters
    ----------
    same as TimeSeries class"""
    _type = 'snIIpl'
    _available_models = ut.Templatelist_fromsncosmo('sniipl')


class SNIIb(TimeSeries):
    """SNIIb class.

    Parameters
    ----------
   same as TimeSeries class   """
    _type = 'snIIb'
    _available_models = ut.Templatelist_fromsncosmo('sniib')

class SNIIn(TimeSeries):
    """SNIIn class.

    Parameters
    ----------
   same as TimeSeries class   """
    _type = 'snIIn'
    _available_models = ut.Templatelist_fromsncosmo('sniin')


class SNIbc(TimeSeries):
    """SNIb/c class.

    Parameters
    ----------
    same as TimeSeries class"""
    _type = 'snIb/c'
    _available_models = ut.Templatelist_fromsncosmo('snib/c')

class SNIc(TimeSeries):
    """SNIIn class.

    Parameters
    ----------
   same as TimeSeries class   """
    _type = 'snIc'
    _available_models = ut.Templatelist_fromsncosmo('snic')


class SNIb(TimeSeries):
    """SNIIn class.

    Parameters
    ----------
   same as TimeSeries class   """
    _type = 'snIb'
    _available_models = ut.Templatelist_fromsncosmo('snib')


class SNIc_BL(TimeSeries):
    """SNIIn class.

    Parameters
    ----------
   same as TimeSeries class   """
    _type = 'snIc-BL'
    _available_models = ut.Templatelist_fromsncosmo('snic-bl')
