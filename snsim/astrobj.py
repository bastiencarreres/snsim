"""Contains transients models."""

import copy
import abc
import numpy as np
import pandas as pd
import sncosmo as snc
from .constants import C_LIGHT_KMS
from . import utils as ut


class AstrObj(abc.ABC):
    """Basic class for transients.

    Parameters
    ----------
    sim_par: dict
        Simulation parameters.
    effect: list(snc.PropagationEffect)
        Effects to apply to the model.
        

    | sim_par
    | ├── zcos, cosmological redshift
    | ├── zpcmb, CMB dipole redshift contribution
    | ├── como_dist, comoving distance of the obj
    | ├── vpec, obj peculiar velocity
    | ├── ra, obj Right Ascension
    | ├── dec, obj Declinaison
    | └── t0, obj peak time
    |  sncosmo
    """

    _type = ''
    _base_attrs = ['ID', 'ra', 'dec', 
                   'zcos', 'vpec', 'zpcmb', 
                   'como_dist', 'template']
    
    _obj_attrs = ['']
    _available_models = ['']

    def __init__(self, source, sim_par, effects=[]):
        
        # -- Copy input parameters dic
        self._sim_par = copy.copy(sim_par)
        
        if 'ID' not in self._sim_par:
            self._sim_par['ID'] = 0
        
        # -- Add template name
        if source.name not in self._available_models:
            raise ValueError(f'{source.name} not available.')
        self._sim_par['template'] = source.name
        
        # -- Update attrs
        for k in self._base_attrs:
            setattr(self, k, self._sim_par[k])

        # -- sncosmo model
        self._sim_model = self._init_model(source, effects)
        
        # -- Update attr of astrobj class
        for k in self._obj_attrs:
            setattr(self, k, self._sim_par[k])
        
        #self._check_keys()
        
    def _check_keys(self):
        for k in self.sim_model.param_names:
            if (k!='z') & (k not in self.sim_par):
                print(f'{k} is not in sim_par, it will be set to default value')   
    
    @abc.abstractmethod
    def _set_model_par(self, model):
        """This method set model parameters that are not t0 or z."""
        pass
        
    def _init_model(self, source, effects):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.

        """

        model = snc.Model(
            source=source,
            effects=[eff['source'] for eff in effects],
            effect_names=[eff['name'] for eff in effects],
            effect_frames=[eff['frame'] for eff in effects])
        
        for eff in effects:
            for k in eff['source'].param_names:
                if k in self._sim_par:
                    model.set(self._sim_par[eff['name'] + k])
        
        model.set(
            t0=self._sim_par['t0'], 
            z=self.zobs)
        
        model = self._set_model_par(model)

        return model
        
    # def _init_dust(self):
    #     """Initialise dust."""
    #     if self._sim_par['mw'] is not None:
    #         if 'rv' not in self.mw_dust:
    #             self.mw_dust['rv'] = 3.1
    #         for model in self.sim_model.values():
    #             dst_ut.init_mw_dust(model, self.mw_dust)

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

        # mask to delete observed points outside time range of the model
        obs = obs[(obs['time']  > self.sim_model.mintime()) & (obs['time'] < self.sim_model.maxtime())]

        if self._sim_par['mod_fcov']:
            # -- Implement the flux variation due to simulation model covariance
            gen = np.random.default_rng(random_seeds[0])
            fluxtrue, fluxcov = self.sim_model.bandfluxcov(
                obs['band'],
                obs['time'],
                zp=obs['zp'],
                zpsys=obs['zpsys'])

            fluxtrue += gen.multivariate_normal(np.zeros(len(fluxcov)),
                                                fluxcov,
                                                check_valid='ignore',
                                                method='eigh')

        else:
            fluxtrue = self.sim_model.bandflux(
                obs['band'],
                obs['time'],
                zp=obs['zp'],
                zpsys=obs['zpsys'])

        # -- Noise computation : Poisson Noise + Skynoise + ZP noise
        fluxerrtrue = np.sqrt(
            np.abs(fluxtrue) / obs.gain
            + obs.skynoise**2
            + (np.log(10) / 2.5 * fluxtrue * obs.sig_zp)**2)

        gen = np.random.default_rng(random_seeds[1])
        flux = fluxtrue + gen.normal(loc=0., scale=fluxerrtrue)
        fluxerr = np.sqrt(
            np.abs(flux) / obs.gain
            + obs.skynoise**2
            + (np.log(10) / 2.5 * flux * obs.sig_zp)**2)

        # -- Set magnitude
        mag = np.zeros_like(flux)
        magerr = np.zeros_like(flux)

        positive_fmask = flux > 0
        flux_pos = flux[positive_fmask]

        mag[positive_fmask] = -2.5 * np.log10(flux_pos) + obs['zp'][positive_fmask]

        magerr[positive_fmask] = 2.5 / np.log(10) * 1 / flux_pos * fluxerr[positive_fmask]

        mag[~positive_fmask] = np.nan
        magerr[~positive_fmask] = np.nan

        # -- Create DataFrame of the lightcurve
        sim_lc = pd.DataFrame({
            'time': obs['time'],
            'fluxtrue': fluxtrue,
            'fluxerrtrue': fluxerrtrue,
            'flux': flux,
            'fluxerr': fluxerr,
            'mag': mag,
            'magerr': magerr,
            'zp': obs['zp'],
            'zpsys': obs['zpsys'],
            'gain': obs['gain'],
            'skynoise': obs['skynoise']
                            })

        # TODO - BC: Maybe remove this "for loop"
        for k in obs.columns:
            if k not in sim_lc.columns:
                sim_lc[k] = obs[k].values

        sim_lc.attrs = {'mu': self.mu,
                        'zobs': self.zobs,
                        'zCMB': self.zCMB,
                        **self._sim_par}

        sim_lc.reset_index(inplace=True, drop=True)
        sim_lc.index.set_names('epochs', inplace=True)
        return sim_lc

    @property
    def zpec(self):
        """Get peculiar velocity redshift."""
        return self.vpec / C_LIGHT_KMS

    @property
    def zCMB(self):
        """Get CMB frame redshift."""
        return (1 + self.zcos) * (1 + self.zpec) - 1.

    @property
    def zobs(self):
        """Get observed redshift."""
        return (1 + self.zcos) * (1 + self.zpec) * (1 + self.zpcmb) - 1.

    @property
    def mu(self):
        """Get distance moduli."""
        return 5 * np.log10((1 + self.zcos) * (1 + self.zpcmb) *
                            (1 + self.zpec)**2 * self.como_dist) + 25

    @property
    def sim_model(self):
        return self._sim_model

    @property
    def sim_par(self):
        return self._sim_par


class SNIa(AstrObj):
    """SNIa class.

    Parameters
    ----------
    sim_par : dict
        Parameters of the SN.
    | same as AstrObj parameters
    | └── mag_sct, coherent mag scattering.
    sim_model : sncosmo.Model
        sncosmo Model to use.

    | same as AstrObj model_par 
    | ├── M0, SNIa absolute magnitude
    | ├── sigM, sigma of coherent scattering
    | └── used model parameters
    """
    _type = 'snIa'
    _available_models = ['salt2', 'salt3']
    _obj_attrs = ['M0', 'mb', 'mag_sct']

    def _set_model_par(self, model):
        """Set SN Ia parameters to sncosmo model.

        Notes
        -----
        Set attributes dependant on SN model
        """
        M0 = self._sim_par['M0'] + self._sim_par['mag_sct']
        
        if self._sim_par['relation'].lower() == 'salttripp':
            if model.source.name not in ['salt2', 'salt3']:
                raise ValueError('SALTTripp only available for salt2 & salt3 models')
            
            self._obj_attrs.extend(['alpha', 'beta', 'x0', 'x1', 'c'])
            
            # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
            # beta*c : scattering due to color and stretch} + {coherent intrinsic scattering}
            self._sim_par['mb'] = self.SALTTripp(
                M0, 
                self._sim_par['alpha'], 
                self._sim_par['beta'], 
                self._sim_par['x1'], 
                self._sim_par['c']) + self.mu 
            
            # Compute the x0 parameter
            model.set(
                x1=self._sim_par['x1'], 
                c=self._sim_par['c'])
            
            model.set_source_peakmag(self._sim_par['mb'], 'bessellb', 'ab')
            self._sim_par['x0'] = model.get('x0')
        else:
            # TODO - BC : Find a way to use lambda function for relation
            raise ValueError('Relation not available')
        return model
    
    @staticmethod
    def SALTTripp(M0, alpha, beta, x1, c):
        return M0  + alpha * x1 - beta * c


class TimeSeries(AstrObj):
    """TimeSeries class.

    Parameters
    ----------
    sn_par : dict
        Parameters of the object.

    | same as AstrObj parameters
    | └── mag_sct, coherent mag scattering.
    sim_model : sncosmo.Model
        sncosmo Model to use.
    model_par : dict
        General model parameters.

    | same as AstrObj model_par
    | ├── M0,  absolute magnitude
    | ├── sigM, sigma of coherent scattering
    | └── used model parameters
    """
    _obj_attrs = ['amplitude', 'mb', 'mag_sct']

    def _set_model_par(self, model):
        """Extract and compute SN parameters that depends on used model.

        Notes
        -----
        Set attributes dependant on SN model
            - mb -> self.mb
            - amplitude -> self.sim_amplitude
            - Template -> self._params['template']  SED template used
        """
        M0 = self._sim_par['M0'] + self._sim_par['mag_sct']

        m_r = self.mu + M0
            
        # Compute the amplitude  parameter
        model.set_source_peakmag(m_r, 'bessellr', 'ab')
        self._sim_par['mb'] = model.source_peakmag('bessellb', 'ab')
        self._sim_par['amplitude'] = model.get('amplitude')
        return model


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
    same as TimeSeries class."""
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
