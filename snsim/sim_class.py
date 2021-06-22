"""This module contains the class which are used in the simulation"""

import sqlite3
import numpy as np
import sncosmo as snc
from astropy.table import Table
from astropy.io import fits
import pandas as pd
from . import utils as ut
from . import salt_utils as salt_ut
from .constants import C_LIGHT_KMS
from . import scatter as sct
from . import nb_fun as nbf


class SN:
    """This class represent SN object.

    Parameters
    ----------
    sn_par : dict
        Contains intrinsic SN parameters generate by SNGen.
        snpar
        ├── zcos
        ├── como_dist
        ├── z2cmb
        ├── sim_t0
        ├── ra
        ├── dec
        ├── vpec
        └── mag_smear
    sim_model : sncosmo.Model
        The sncosmo model used to generate the SN ligthcurve.
    model_par : dict
        Contains general model parameters and sncsomo parameters.
        model_par
        ├── M0
        ├── SN model general parameters
        └── sncosmo
            └── SN model parameters needed by sncosmo

    Attributes
    ----------
    _sn_par : dic
        Contains the sn_par parameters.
    _epochs : Astropy Table
        Contains the epochs when the SN is observed by the survey.
    sim_lc : Astropy Table
        The result of the SN ligthcurve simulation with sncosmo.
    _ID : type
        The Supernovae ID.
    sim_model : sncosmo.Model
        A copy of the input sncosmo Model.
    _model_par :
        A copy of input model_par with some model dependents quantities added.

    Methods
    -------
    _init_model_par(self)
        Init model parameters of the SN use dto compute mu.
    _reformat_sim_table(self)
        Give the good format to sncosmo output Table.
    pass_cut(self, nep_cut)
        Test if the SN pass the cuts given in nep_cut.
    gen_flux(self)
        Generate the ligthcurve flux with sncosmo.
    get_lc_hdu(self)
        Give a hdu version of the ligthcurve table
    """

    def __init__(self, sn_par, sim_model, model_par):
        self.sim_model = sim_model.__copy__()
        self._sn_par = sn_par
        self._model_par = model_par
        self._init_model_par()
        self._epochs = None
        self._sim_lc = None
        self._ID = None

    @property
    def ID(self):
        """Get SN ID"""
        return self._ID

    @ID.setter
    def ID(self, ID):
        """Set SN ID"""
        if isinstance(ID, int):
            self._ID = ID
        else:
            print('SN ID must be an integer')
        if self.sim_lc is not None:
            self.sim_lc.meta['sn_id'] = self._ID

    @property
    def sim_t0(self):
        """Get SN peakmag time"""
        return self._sn_par['sim_t0']

    @property
    def vpec(self):
        """Get SN peculiar velocity"""
        return self._sn_par['vpec']

    @property
    def zcos(self):
        """Get SN cosmological redshift"""
        return self._sn_par['zcos']

    @property
    def como_dist(self):
        """Get SN comoving distance"""
        return self._sn_par['como_dist']

    @property
    def coord(self):
        """Get SN coordinates (ra,dec)"""
        return self._sn_par['ra'], self._sn_par['dec']

    @property
    def mag_smear(self):
        """Get SN coherent scattering term"""
        return self._sn_par['mag_smear']

    @property
    def zpec(self):
        """Get SN peculiar velocity redshift"""
        return self.vpec / C_LIGHT_KMS

    @property
    def zCMB(self):
        """Get SN CMB frame redshift"""
        return (1 + self.zcos) * (1 + self.zpec) - 1.

    @property
    def z2cmb(self):
        """Get SN redshift due to our motion relative to CMB"""
        return self._sn_par['z2cmb']

    @property
    def z(self):
        """Get SN observed redshift"""
        return (1 + self.zcos) * (1 + self.zpec) * (1 + self.z2cmb) - 1.

    @property
    def epochs(self):
        """Get SN observed redshift"""
        return self._epochs

    @epochs.setter
    def epochs(self, ep_dic):
        """Get SN observed epochs"""
        self._epochs = ep_dic

    @property
    def sim_mu(self):
        """Get SN distance moduli"""
        return 5 * np.log10((1 + self.zcos) * (1 + self.z2cmb) *
                            (1 + self.zpec)**2 * self.como_dist) + 25

    @property
    def smear_mod_seed(self):
        """Get SN  smear model if exist"""
        if 'G10_RndS' in self._model_par['sncosmo']:
            return self._model_par['sncosmo']['G10_RndS']
        elif 'C11_RndS' in self._model_par['sncosmo']:
            return self._model_par['sncosmo']['C11_RndS']
        else:
            return None

    @property
    def sim_lc(self):
        """Get sim_lc"""
        return self._sim_lc

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

        M0 = self._model_par['M0']
        if self.sim_model.source.name in ['salt2', 'salt3']:
            # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
            # beta*c : scattering due to color and stretch} + {intrinsic smearing}
            alpha = self._model_par['alpha']
            beta = self._model_par['beta']
            x1 = self._model_par['sncosmo']['x1']
            c = self._model_par['sncosmo']['c']
            mb = self.sim_mu + M0 - alpha * \
                x1 + beta * c + self.mag_smear

            x0 = salt_ut.mB_to_x0(mb)
            self.sim_mb = mb
            self.sim_x0 = x0
            self.sim_x1 = x1
            self.sim_c = c
            self._model_par['sncosmo']['x0'] = x0

        # elif self.sim_model.source.name == 'snoopy':
            # TODO

    def pass_cut(self, nep_cut):
        """Check if the SN pass the given cuts.

        Parameters
        ----------
        nep_cut : list
            nep_cut = [[nep_min1,Tmin,Tmax],[nep_min2,Tmin2,Tmax2,'filter1'],...].

        Returns
        -------
        boolean
            True or False.

        """
        if self.epochs is None:
            return False
        else:
            for cut in nep_cut:
                cutMin_obsfrm, cutMax_obsfrm = cut[1] * (1 + self.z), cut[2] * (1 + self.z)
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
        Set the sim_lc attribute as an astropy Table
        """

        params = {**{'z': self.z, 't0': self.sim_t0}, **self._model_par['sncosmo']}
        self._sim_lc = snc.realize_lcs(self.epochs, self.sim_model, [params], scatter=False)[0]

        self._sim_lc['fluxerr'] = np.sqrt(self.sim_lc['fluxerr']**2 +
                                          (np.log(10)/2.5 * self.sim_lc['flux'] *
                                           self.epochs['sig_zp'])**2)

        self._sim_lc['flux'] = rand_gen.normal(loc=self.sim_lc['flux'],
                                               scale=self.sim_lc['fluxerr'])

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
        not_to_change = ['G10', 'C11']
        for k in self.epochs.keys():
            if k not in self.sim_lc.copy().keys():
                self._sim_lc[k] = self.epochs[k].copy()

        for k in self.sim_lc.meta.copy():
            if k != 'z' and k[:3] not in not_to_change:
                self.sim_lc.meta['sim_' + k] = self.sim_lc.meta.pop(k)

        if self.ID is not None:
            self.sim_lc.meta['sn_id'] = self.ID

        self.sim_lc.meta['vpec'] = self.vpec
        self.sim_lc.meta['zcos'] = self.zcos
        self.sim_lc.meta['zpec'] = self.zpec
        self.sim_lc.meta['z2cmb'] = self.z2cmb
        self.sim_lc.meta['zCMB'] = self.zCMB
        self.sim_lc.meta['ra'] = self.coord[0]
        self.sim_lc.meta['dec'] = self.coord[1]
        self.sim_lc.meta['sim_mb'] = self.sim_mb
        self.sim_lc.meta['sim_mu'] = self.sim_mu
        self.sim_lc.meta['m_smear'] = self.mag_smear

    def get_lc_hdu(self):
        """Convert the astropy Table to a hdu.

        Returns
        -------
        fits hdu
            A hdu object containing the sim_lc Table.

        """
        return fits.table_to_hdu(self.sim_lc)


class SnGen:
    """This class set up the random part of the SN simulation.

    Parameters
    ----------
    sn_int_par : dict
        Intrinsic parameters of the supernovae.
        sn_int_par
        ├── M0 # Standard absolute magnitude
        ├── mag_smear # Coherent intrinsic scattering
        └── smear_mod # Wavelenght dependant smearing (Optional)
    model_config : dict
        The parameters of the sn simulation model to use.
        model_config
        ├── model_dir # The directory of the model file
        ├── model_name # The name of the model
        └── model parameters # All model needed parameters
    cmb : dict
        The cmb parameters
        cmb
        ├── vcmb
        ├── l_cmb
        └── b_cmb
    cosmology : astropy.cosmology
        The astropy cosmological model to use.
    vpec_dist : dict
        The parameters of the peculiar velocity distribution.
        vpec_dist
        ├── mean_vpec
        └── sig_vpec
    host : class SnHost
        The host class to introduce sn host.

    Attributes
    ----------
    _sn_int_par : dict
        A copy of the input sn_int_par dict.
    _model_config : dict
        A copy of the input model_config dict.
    _cmb : dict
        A copy of the input cmb dict.
    sim_model : sncosmo.Model
        The model used to simulate supernovae
    _model_keys : dict
        SN model global parameters names.
    _vpec_dist : dict
        A copy of the input vpec_dist dict.
    _cosmology : astropy.cosmology
        A copy of the input cosmology model.
    _host : class SnHost
        A copy of the input SnHost class.
    _time_range : list(float,float)
        The range of time in which simulate the peak.
    _z_cdf : list(numpy.ndaray(float), numpy.ndaray(float))
        The cumulative distribution of the redshifts.

    Methods
    -------
    __init__(self, sim_par, host=None)
        Initialise the SNGen object.
    _init_model_keys(self)
        Init the SN model parameters names.
    _init_sim_model(self)
        Configure the sncosmo Model
    __call__(self, n_sn, z_range, time_range, rand_seed)
        Simulate a given number of sn in a given redshift range and time range
        using the given random seed.
    gen_peak_time(self, n, rand_seed)
        Randomly generate peak time in the given time range.
    gen_coord(self, n, rand_seed)
        Generate ra, dec uniformly on the sky.
    gen_zcos(self, n, rand_seed)
        Generate redshift following a distribution.
    gen_model_par(self, n, rand_seed)
        Generate the random parameters of the sncosmo model.
    gen_salt_par(self, n, rand_seed)
        Generate the parameters for the SALT2/3 model.
    gen_vpec(self, n, rand_seed)
        Generate peculiar velocities on a gaussian law.
    gen_coh_scatter(self, n, rand_seed)
        Generate the coherent scattering term.
    _gen_noise_rand_seed(self, n, rand_seed)
        Generate the rand seeds for random fluxerror.
    """

    def __init__(self, sn_int_par, model_config, cmb, cosmology, vpec_dist, host=None):
        self._sn_int_par = sn_int_par
        self._model_config = model_config
        self._cmb = cmb
        self.sim_model = self._init_sim_model()
        self._model_keys = self._init_model_keys()
        self._vpec_dist = vpec_dist
        self._cosmology = cosmology
        self._host = host
        self._time_range = None
        self._z_cdf = None

    @property
    def model_config(self):
        """Get sncosmo model parameters"""
        return self._model_config

    @property
    def host(self):
        """Get the host class"""
        return self._host

    @property
    def vpec_dist(self):
        """Get the peculiar velocity distribution parameters"""
        return self._vpec_dist

    @property
    def sn_int_par(self):
        """Get sncosmo configuration parameters"""
        return self._sn_int_par

    @property
    def cosmology(self):
        """Get astropy cosmological model"""
        return self._cosmology

    @property
    def cmb(self):
        """Get cmb used parameters"""
        return self._cmb

    @property
    def snc_model_time(self):
        """Get the sncosmo model mintime and maxtime"""
        return self.sim_model.mintime(), self.sim_model.maxtime()

    @property
    def time_range(self):
        """Get time range"""
        return self._time_range

    @time_range.setter
    def time_range(self, time_range):
        """Set the time range"""
        if time_range[0] > time_range[1]:
            print('Time range should be [Tmin,Tmax]')
        else:
            self._time_range = time_range

    @property
    def z_cdf(self):
        """Get the redshift cumulative distribution"""
        return self._z_cdf

    @z_cdf.setter
    def z_cdf(self, cdf):
        """Set the redshift cumulative distribution"""
        self._z_cdf = cdf

    def _init_sim_model(self):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.

        """
        model = ut.init_sn_model(self.model_config['model_name'],
                                 self.model_config['model_dir'])

        if 'smear_mod' in self.sn_int_par:
            model = sct.init_sn_smear_model(model, self.sn_int_par['smear_mod'])
        return model

    def _init_model_keys(self):
        """Initialise the model keys depends on the SN simulation model.

        Returns
        -------
        list
            A dict containing all the usefull keys of the SN model.
        """
        model_name = self.model_config['model_name']
        if model_name in ('salt2', 'salt3'):
            model_keys = ['alpha', 'beta']
        return model_keys

    def __call__(self, n_sn, rand_gen=None):
        """Launch the simulation of SN.

        Parameters
        ----------
        n_sn : int
            Number of SN to simulate.
        z_range : list(float)
            The redshift range of simulation -> 2 elmt zmin and zmax.
        rand_seed : int
            The random seed of the simulation.

        Returns
        -------
        list(SN)
            A list containing SN object.

        """
        # -- RANDOM PART :
        # ---- Order is important for reproduce a simulation
        # ---- because of the numpy random generator object
        # ---- The order is : 1. t0 -> SN peak time
        # ----                2. zcos -> SN cosmological redshifts
        # ----                3. mag_smear -> Coherent magnitude scattering
        # ----                4. opt_seeds -> 3 indep randseeds for coord, vpec and model

        if rand_gen is None:
            rand_gen = np.random.default_rng()

        # -- Generate peak magnitude
        t0 = self.gen_peak_time(n_sn, rand_gen)

        # -- Generate cosmological redshifts
        zcos = self.gen_zcos(n_sn, rand_gen)

        # -- Generate coherent mag smearing
        mag_smear = self.gen_coh_scatter(n_sn, rand_gen)

        # -- Generate 3 randseeds for optionnal parameters randomization
        opt_seeds = rand_gen.integers(low=1000, high=100000, size=3)

        # - Generate random parameters dependants on sn model used
        rand_model_par = self.gen_model_par(n_sn, np.random.default_rng(opt_seeds[0]))

        # -- If there is host use them
        if self.host is not None:
            treshold = (self.z_cdf[0][-1] - self.z_cdf[0][0])/100
            host = self.host.host_near_z(zcos, treshold)
            ra = host['ra']
            dec = host['dec']
            zcos = host['redshift']
            vpec = host['vp_sight']
        else:
            ra, dec = self.gen_coord(n_sn, np.random.default_rng(opt_seeds[1]))
            vpec = self.gen_vpec(n_sn, np.random.default_rng(opt_seeds[2]))

        # -- SN initialisation part :

        sn_par = ({'zcos': z,
                   'como_dist': self.cosmology.comoving_distance(z).value,
                   'z2cmb': ut.compute_z2cmb(r, d, self.cmb),
                   'sim_t0': t,
                   'ra': r,
                   'dec': d,
                   'vpec': v,
                   'mag_smear': ms
                   } for z, t, r, d, v, ms in zip(zcos, t0, ra, dec, vpec, mag_smear))

        model_default = {'M0': self.sn_int_par['M0']}
        for k in self._model_keys:
            model_default[k] = self.model_config[k]

        model_par_list = ({**model_default, 'sncosmo': mpsn} for mpsn in rand_model_par)

        return [SN(snp, self.sim_model, mp) for snp, mp in zip(sn_par, model_par_list)]

    def gen_peak_time(self, n, rand_gen):
        """Generate uniformly n peak time in the survey time range.

        Parameters
        ----------
        n : int
            Number of time to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float)
            A numpy array which contains generated peak time.

        """
        t0 = rand_gen.uniform(*self.time_range, size=n)
        return t0

    @staticmethod
    def gen_coord(n, rand_gen):
        """Generate n coords (ra,dec) uniformly on the sky sphere.

        Parameters
        ----------
        n : int
            Number of coord to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing generated coordinates.

        """
        ra = rand_gen.uniform(low=0, high=2 * np.pi, size=n)
        dec_uni = rand_gen.random(size=n)
        dec = np.arcsin(2 * dec_uni - 1)
        return ra, dec

    def gen_zcos(self, n, rand_gen):
        """Generate n cosmological redshift in a range.

        Parameters
        ----------
        n : int
            Number of redshift to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float)
            A numpy array which contains generated cosmological redshift.
        """
        uni_var = rand_gen.random(size=n)
        zcos = np.interp(uni_var, self.z_cdf[1], self.z_cdf[0])
        return zcos

    def gen_model_par(self, n, rand_gen):
        """Generate model dependant parameters.

        Parameters
        ----------
        n : int
            Number of parameters to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        dict
            One dictionnary containing 'parameters names': numpy.ndaray(float).

        """
        model_name = self.model_config['model_name']

        if model_name in ('salt2', 'salt3'):
            sim_x1, sim_c = self.gen_salt_par(n, rand_gen)
            model_par_sncosmo = [{'x1': x1, 'c': c} for x1, c in zip(sim_x1, sim_c)]

        if 'G10_' in self.sim_model.effect_names:
            seeds = rand_gen.integers(low=1000, high=100000, size=n)
            for par, s in zip(model_par_sncosmo, seeds):
                par['G10_RndS'] = s

        elif 'C11_' in self.sim_model.effect_names:
            seeds = rand_gen.integers(low=1000, high=100000, size=n)
            for par, s in zip(model_par_sncosmo, seeds):
                par['C11_RndS'] = s

        return model_par_sncosmo

    def gen_salt_par(self, n, rand_gen):
        """Generate n SALT parameters.

        Parameters
        ----------
        n : int
            Number of parameters to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing SALT2 x1 and c generated parameters.

        """
        sig_x1_low, sig_x1_high = ut.is_asym(self.model_config['sig_x1'])
        sig_c_low, sig_c_high = ut.is_asym(self.model_config['sig_c'])

        sim_x1 = [ut.asym_gauss(self.model_config['mean_x1'],
                                sig_x1_low,
                                sig_x1_high,
                                rand_gen) for i in range(n)]

        sim_c = [ut.asym_gauss(self.model_config['mean_c'],
                               sig_c_low,
                               sig_c_high,
                               rand_gen) for i in range(n)]

        return sim_x1, sim_c

    def gen_vpec(self, n, rand_gen):
        """Generate n peculiar velocities.

        Parameters
        ----------
        n : int
            Number of vpec to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing vpec (km/s) generated.

        """
        vpec = rand_gen.normal(
            loc=self.vpec_dist['mean_vpec'],
            scale=self.vpec_dist['sig_vpec'],
            size=n)
        return vpec

    def gen_coh_scatter(self, n, rand_gen):
        """Generate n coherent mag smear term.

        Parameters
        ----------
        n : int
            Number of mag smear terms to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing smear terms generated.

        """
        mag_smear = rand_gen.normal(
            loc=0, scale=self.sn_int_par['mag_smear'], size=n)
        return mag_smear


class SurveyObs:
    """This class deals with the observations of the survey.

    Parameters
    ----------
    survey_config : dic
        It contains all the survey configuration.
        survey_config
        ├── survey_file PATH TO SURVEY FILE
        ├── ra_size RA FIELD SIZE IN DEG -> float
        ├── dec_size DEC FIELD SIZE IN DEG -> float
        ├── gain CCD GAIN e-/ADU -> float
        ├── start_day STARTING DAY -> float or str, opt
        ├── end_day ENDING DAY -> float or str, opt
        ├── duration SURVEY DURATION -> float, opt
        ├── zp FIXED ZEROPOINT -> float, opt
        ├── survey_cut, CUT ON DB FILE -> dict, opt
        └── add_data, LIST OF KEY TO ADD METADATA -> list(str), opt

    Attributes
    ----------
    _survey_prop : dict
        Copy of the survey_prop input dict.
    obs_table : astropy.Table
        Table containing the observation.
    _db_cut : dict
        Copy of the db_cut input
    _db_file : str
        Copy of the db_file input
    _band_dic : str
        Copy of the band_dic input
    _add_keys : dict
        Copy of the add_keys input

    Methods
    -------
    _init_field_dic(self):
        Create a dictionnary with fieldID and coord.

    _extract_from_db(self)
        Extract the observation from SQL data base.

     _read_start_end_days(self):
        Initialise the start and ending day from survey configuration.

    epochs_selection(self, SN)
        Give the epochs of observation of a given SN.

     _make_obs_table(self, epochs_selec):
        Create the astropy table from selection bool array.
    """

    def __init__(self, survey_config):
        self._config = survey_config
        self._obs_table, self._start_end_days = self._extract_from_db()
        self._field_dic = self._init_field_dic()

    def _init_field_dic(self):
        """Create a dictionnary with fieldID and coord.

        Returns
        -------
        dict
            fieldID : {'ra' : fieldRA, 'dec': fieldDec}.

        """

        field_list = self.obs_table['fieldID'].unique()
        dic = {}
        for f in field_list:
            idx = nbf.find_first(f, self.obs_table['fieldID'].values)
            dic[f] = {'ra': self.obs_table['fieldRA'][idx],
                      'dec': self.obs_table['fieldDec'][idx]}
        return dic

    @property
    def config(self):
        """Survey configuration"""
        return self._config

    @property
    def band_dic(self):
        """Get the dic band_survey : band_sncosmo"""
        return self.config['band_dic']

    @property
    def obs_table(self):
        """Table of the observations"""
        return self._obs_table

    @property
    def field_size(self):
        """Get field size (ra,dec) in radians"""
        ra_size_rad = np.radians(self._config['ra_size'])
        dec_size_rad = np.radians(self._config['dec_size'])
        return ra_size_rad, dec_size_rad

    @property
    def gain(self):
        """Get CCD gain in e-/ADU"""
        return self._config['gain']

    @property
    def zp(self):
        """Get zero point and it's uncertainty"""
        if 'zp' in self._config:
            zp = self._config['zp']
        else:
            zp = 'zp_in_obs'
        if 'sig_zp' in self._config:
            sig_zp = self._config['sig_zp']
        else:
            sig_zp = 'sig_zp_in_obs'
        return (zp, sig_zp)

    @property
    def sig_psf(self):
        """Get PSF width"""
        if 'sig_psf' in self._config:
            sig_psf = self._config['zp']
        else:
            sig_psf = 'psf_in_obs'
        return sig_psf

    @property
    def duration(self):
        """Get the survey duration in days"""
        duration = self.start_end_days[1].mjd - self.start_end_days[0].mjd
        return duration

    @property
    def start_end_days(self):
        """Get the survey start and ending days"""
        return self._start_end_days[0], self._start_end_days[1]

    def _read_start_end_days(self, obs_dic):
        """Initialise the start and ending day from survey configuration.

        Parameters
        ----------
        obs_dic : pandas.DataFrame
            The actual obs_dic to take min and max obs date if not given.

        Returns
        -------
        tuple(astropy.time.Time)
            astropy Time object of the starting and the ending day of the survey.

        Notes
        -----
        The final starting and ending days of the survey may differ from the input
        because the survey file maybe not contain exactly observation on the input
        day.

        Note that end_day key has priority on duration
        """

        if 'start_day' in self.config:
            start_day = self.config['start_day']
        else:
            start_day = obs_dic['expMJD'].min()

        start_day = ut.init_astropy_time(start_day)

        if 'end_day' in self.config:
            end_day = self.config['end_day']
        elif 'duration' in self.config:
            end_day = start_day.mjd + self.config['duration']
        else:
            end_day = obs_dic['expMJD'].max()

        end_day = ut.init_astropy_time(end_day)

        return start_day, end_day

    def _extract_from_db(self):
        """Extract the observations table from SQL data base.

        Returns
        -------
        pandas.DataFrame
            The observations table.
        tuple(astropy.time.Time)
            The starting time and ending time of the survey.
        """

        con = sqlite3.connect(self._config['survey_file'])

        keys = ['expMJD',
                'filter',
                'fieldID',
                'fieldRA',
                'fieldDec',
                'fiveSigmaDepth']

        if not 'zp' in self.config:
            keys += ['zp']

        if not 'sig_zp' in self.config:
            keys += ['sig_zp']

        if not 'sig_psf' in self.config:
            keys += ['FWHMeff']

        if 'add_data' in self.config:
            add_k = (k for k in self.config['add_data'] if k not in keys)
            keys += add_k

        where = ''
        if 'survey_cut' in self.config:
            cut_dic = self.config['survey_cut']
            where = " WHERE "
            for cut_var in cut_dic:
                where += "("
                for cut in cut_dic[cut_var]:
                    cut_str = f"{cut}"
                    where += f"{cut_var}{cut_str} OR "
                where = where[:-4]
                where += ") AND "
            where = where[:-5]
        query = 'SELECT '
        for k in keys:
            query += k+','
        query = query[:-1]
        query += ' FROM Summary' + where + ';'
        obs_dic = pd.read_sql_query(query, con)

        # avoid crash on errors
        obs_dic.query('fiveSigmaDepth > 0', inplace=True)

        start_day_input, end_day_input = self._read_start_end_days(obs_dic)

        if start_day_input.mjd < obs_dic['expMJD'].min():
            raise ValueError('start_day before first day in survey file')
        elif end_day_input.mjd > obs_dic['expMJD'].max():
            raise ValueError('end_day after last day in survey file')

        obs_dic.query(f"expMJD >= {start_day_input.mjd} & expMJD <= {end_day_input.mjd}",
                      inplace=True)
        obs_dic.reset_index(drop=True, inplace=True)

        if obs_dic.size == 0:
            raise RuntimeError('No observation for the given survey start_day and duration')
        start_day = ut.init_astropy_time(obs_dic['expMJD'].min())
        end_day = ut.init_astropy_time(obs_dic['expMJD'].max())
        return obs_dic, (start_day, end_day)

    def epochs_selection(self, SN_obj):
        """Give the epochs of observations of a given SN.

        Parameters
        ----------
        SN : SN object
            A class SN object.

        Returns
        -------
        astropy.Table
            astropy table containing the SN observations.

        """

        ModelMinT_obsfrm = SN_obj.sim_model.mintime() * (1 + SN_obj.z)
        ModelMaxT_obsfrm = SN_obj.sim_model.maxtime() * (1 + SN_obj.z)
        ra, dec = SN_obj.coord

        # time selection
        epochs_selec, selec_fields_ID = nbf.time_selec(self.obs_table['expMJD'].values,
                                                       SN_obj.sim_t0,
                                                       ModelMaxT_obsfrm,
                                                       ModelMinT_obsfrm,
                                                       self.obs_table['fieldID'].values)

        ra_fields = np.array(list(map(lambda x: x['ra'],
                                      map(self._field_dic.get, selec_fields_ID))))
        dec_fields = np.array(list(map(lambda x: x['dec'],
                                       map(self._field_dic.get, selec_fields_ID))))

        # Compute the coord of the SN in the rest frame of each field
        ra_field_frame, dec_field_frame = ut.change_sph_frame(ra, dec,
                                                              ra_fields,
                                                              dec_fields)

        epochs_selec, is_obs = nbf.is_in_field(epochs_selec,
                                               self._obs_table['fieldID'][epochs_selec].values,
                                               ra_field_frame, dec_field_frame,
                                               self.field_size, selec_fields_ID)
        if is_obs:
            return self._make_obs_table(epochs_selec)
        return None

    def _make_obs_table(self, epochs_selec):
        """ Create the astropy table from selection bool array.

        Parameters
        ----------
        epochs_selec : numpy.ndarray(boolean)
            A boolean array that define the observation selection.

        Returns
        -------
        astropy.Table
            The observations table that correspond to the selection.

        """

        # Capture noise and filter
        mlim5 = self.obs_table['fiveSigmaDepth'][epochs_selec]
        band = self.obs_table['filter'][epochs_selec].astype('U27')

        # Change band name to correpond with sncosmo bands -> CHANGE EMPLACEMENT
        if self.band_dic is not None:
            band = np.array(list(map(self.band_dic.get, band)))

        if self.zp[0] != 'zp_in_obs':
            zp = np.ones(np.sum(epochs_selec)) * self.zp[0]
        else:
            zp = self.obs_table['zp'][epochs_selec]

        if self.zp[1] != 'sig_zp_in_obs':
            sig_zp = np.ones(np.sum(epochs_selec)) * self.zp[1]
        else:
            sig_zp = self.obs_table['sig_zp'][epochs_selec]

        if self.sig_psf != 'psf_in_obs':
            sig_psf = np.ones(np.sum(epochs_selec)) * self.sig_psf
        else:
            sig_psf = self.obs_table['FWHMeff'][epochs_selec]/(2*np.sqrt(2*np.log(2)))

        # Convert maglim to flux noise (ADU) -> skynoise_tot = skynoise * sqrt(4 pi sig_psf**2)
        skynoise = 10.**(0.4 * (zp - mlim5)) / 5

        skynoise[sig_psf > 0] *= np.sqrt(4 * np.pi * sig_psf[sig_psf > 0]**2)

        # Create obs table
        obs = Table({'time': self._obs_table['expMJD'][epochs_selec],
                     'band': band,
                     'gain': [self.gain] * np.sum(epochs_selec),
                     'skynoise': skynoise,
                     'zp': zp,
                     'sig_zp': sig_zp,
                     'zpsys': ['ab'] * np.sum(epochs_selec),
                     'fieldID': self._obs_table['fieldID'][epochs_selec]})

        if 'add_data' in self.config:
            for k in self.config['add_data']:
                if k not in obs:
                    obs[k] = self.obs_table[k][epochs_selec]
        return obs


class SnHost:
    """ Class containing the SN Host parameters.

    Parameters
    ----------
    host_file : str
        fits host file path.
    z_range : list(float)
        The redshift range.

    Attributes
    ----------
    _table : pandas.DataFrame
        Pandas dataframe that contains host data.
    _max_dz : float
        The maximum redshift gap between 2 host.
    _z_range : list(float)
        A copy of input z_range.
    _file
        A copy of input host_file.

    Methods
    -------
    _read_host_file()
        Extract host from host file.
    random_host(n, z_range, rand_seed)
        Random choice of host in a redshift range.

    """

    def __init__(self, host_file, z_range=None):
        self._z_range = z_range
        self._file = host_file
        self._table = self._read_host_file()
        self._max_dz = None

    @property
    def max_dz(self):
        """Get the maximum redshift gap"""
        if self._max_dz is None:
            redshift_copy = np.sort(np.copy(self.table['redshift']))
            diff = redshift_copy[1:] - redshift_copy[:-1]
            self._max_dz = np.max(diff)
        return self._max_dz

    @property
    def table(self):
        """Get astropy Table of host"""
        return self._table

    def _read_host_file(self):
        """Extract host from host file.

        Returns
        -------
        astropy.Table
            astropy Table containing host.

        """
        with fits.open(self._file) as hostf:
            host_list = pd.DataFrame.from_records(hostf[1].data[:])
        host_list = host_list.astype('float64')
        ra_mask = host_list['ra'] < 0
        host_list['ra'][ra_mask] = host_list['ra'][ra_mask] + 2 * np.pi
        if self._z_range is not None:
            z_min, z_max = self._z_range
            return host_list.query(f'redshift >= {z_min} & redshift <= {z_max}')
        return host_list

    def host_near_z(self, z_list, treshold=1e-4):
        """Take the nearest host from a redshift list.

        Parameters
        ----------
        z_list : numpy.ndarray(float)
            The redshifts.
        treshold : float, optional
            The maximum difference tolerance.

        Returns
        -------
        astropy.Table
            astropy Table containing the selected host.

        """
        idx = nbf.find_idx_nearest_elmt(z_list, self.table['redshift'].values, treshold)
        return self.table.iloc[idx]


class SnSimPkl:
    """Class to store simulation as pickle.

    Parameters
    ----------
    sim_lc : list(astropy.Table)
        The simulated lightcurves.
    header : dict
        The metadata of the simulation.

    Attributes
    ----------
    _header : dict
        A copy of input header.
    _sim_lc : list(astropy.Table)
        A copy of input sim_lc.

    """

    def __init__(self, sim_lc, header):
        self._header = header
        self._sim_lc = sim_lc

    @property
    def header(self):
        """Get header"""
        return self._header

    @property
    def sim_lc(self):
        """Get sim_lc"""
        return self._sim_lc

    def get(self, key):
        """Get an array of sim_lc metadata.

        Parameters
        ----------
        key : str
            The metadata to access.

        Returns
        -------
        numpy.ndarray
            The array of the key metadata for all SN.

        """
        return np.array([lc.meta[key] for lc in self.sim_lc])
