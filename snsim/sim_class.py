"""This module contains the class which are used in the simulation"""

import sqlite3
import numpy as np
import sncosmo as snc
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from numpy import power as pw
import pandas as pd
import snsim.utils as ut
from snsim.constants import C_LIGHT_KMS
import snsim.scatter as sct
import snsim.nb_fun as nbf


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
    __init_model_par()
        Init model parameters of the SN use dto compute mu.
    __reformat_sim_table()
        Give the good format to sncosmo output Table.
    pass_cut(nep_cut)
        Test if the SN pass the cuts given in nep_cut.
    gen_flux()
        Generate the ligthcurve flux with sncosmo.
    get_lc_hdu()
        Give a hdu version of the ligthcurve table
    """

    def __init__(self, sn_par, sim_model, model_par):
        self.sim_model = sim_model.__copy__()
        self._sn_par = sn_par
        self._model_par = model_par
        self.__init_model_par()
        self._epochs = None
        self._sim_lc = None
        self._ID = None
        return

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

    def __init_model_par(self):
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

            x0 = ut.mB_to_x0(mb)
            self.sim_mb = mb
            self.sim_x0 = x0
            self.sim_x1 = x1
            self.sim_c = c
            self._model_par['sncosmo']['x0'] = x0

        # elif self.sim_model.source.name == 'snoopy':
            # TODO
        return None

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
        self._sim_lc['flux'] = rand_gen.normal(loc=self.sim_lc['flux'], scale=self.sim_lc['fluxerr'])

        return self.__reformat_sim_table()

    def __reformat_sim_table(self):
        """Give the good format to the sncosmo output Table.

        Returns
        -------
        None

        Notes
        -----
        Directly change the sim_lc attribute

        """
        not_to_change = ['G10','C11']
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
        ├── ra_cmb
        └── dec_cmb
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

    Methods
    -------
    __init__(sim_par, host=None)
        Initialise the SNGen object.
    __call__(n_sn, z_range, time_range, rand_seed)
        Simulate a given number of sn in a given redshift range and time range
        using the given random seed.
    __init_model_keys()
        Init the SN model parameters names.
    __init_sim_model()
        Configure the sncosmo Model
    gen_peak_time(n, time_range, rand_seed)
        Randomly generate peak time in the given time range.
    gen_coord(n, rand_seed)
        Generate ra, dec uniformly on the sky.
    gen_zcos(n, z_range, rand_seed)
        Generate redshift uniformly in a range. #TO CHANGE
    gen_model_par(self, n, rand_seed)
        Generate the random parameters of the sncosmo model.
    gen_salt_par(self, n, rand_seed)
        Generate the parameters for the SALT2/3 model.
    gen_vpec(self, n, rand_seed)
        Generate peculiar velocities on a gaussian law.
    gen_coh_scatter(self, n, rand_seed)
        Generate the coherent scattering term.
    __gen_noise_rand_seed(self, n, rand_seed)
        Generate the rand seeds for random fluxerror.
    """

    def __init__(self, sn_int_par, model_config, cmb, cosmology, vpec_dist, host=None):
        self._sn_int_par = sn_int_par
        self._model_config = model_config
        self._cmb = cmb
        self.sim_model = self.__init_sim_model()
        self._model_keys = self.__init_model_keys()
        self._vpec_dist = vpec_dist
        self._cosmology = cosmology
        self._host = host

    @property
    def model_config(self):
        """Get sncosmo model parameters"""
        return self._model_config

    @property
    def host(self):
        return self._host

    @property
    def vpec_dist(self):
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

    def __init_sim_model(self):
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

    def __init_model_keys(self):
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

    def __call__(self, n_sn, z_range, time_range, rand_gen=None):
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
        if rand_gen is None:
            rand_gen = np.random.default_rng()

        #- Generate peak magnitude
        t0 = self.gen_peak_time(n_sn, time_range, rand_gen)

        #- Generate coherent mag smearing
        mag_smear = self.gen_coh_scatter(n_sn, rand_gen)

        #- Generate random parameters dependants on sn model used
        rand_model_par = self.gen_model_par(n_sn, rand_gen)

        if self.host is not None:
            host = self.host.random_host(n_sn, z_range, rand_gen)
            ra = host['ra']
            dec = host['dec']
            zcos = host['redshift']
            vpec = host['vp_sight']
        else:
            ra, dec = self.gen_coord(n_sn, rand_gen)
            zcos = self.gen_zcos(n_sn, z_range, rand_gen)
            vpec = self.gen_vpec(n_sn, rand_gen)

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

        model_par_list = [{**model_default, 'sncosmo': mpsn} for mpsn in rand_model_par]

        return [SN(snp, self.sim_model, mp) for snp, mp in zip(sn_par, model_par_list)]

    @staticmethod
    def gen_peak_time(n, time_range, rand_gen):
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
        t0 = rand_gen.uniform(*time_range, size=n)
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

    @staticmethod
    def gen_zcos(n, z_range, rand_gen):
        """Generate n cosmological redshift in a range.

        Parameters
        ----------
        n : int
            Number of redshift to generate.
        z_range : list(float)
            The redshift range zmin zmax.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float)
            A numpy array which contains generated cosmological redshift.

        TODO: Generation is uniform, in small shell not really a problem, maybe
        fix this in general
        """
        zcos = rand_gen.uniform(low=z_range[0], high=z_range[1], size=n)
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
        ''' Generate coherent intrinsic scattering '''
        mag_smear = rand_gen.normal(
            loc=0, scale=self.sn_int_par['mag_smear'], size=n)
        return mag_smear

class SurveyObs:
    """This class deals with the observations of the survey.

    Parameters
    ----------
    db_file : str
        Path to the SQL database of observations.
    survey_prop : dict
        The properties of the survey.
    band_dic : dict
        Translate from band name in database to sncosmo band name.
    db_cut : dict
        Selection to add to the SQL query.
    add_keys : list
        database keys to add to observation meta.

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
    __extract_from_db()
        Extract the observation from SQL data base.

    epochs_selection(SN)
        Give the epochs of observation of a given SN.

    """

    def __init__(self, survey_config): #db_file, survey_prop, band_dic=None, survey_cut=None, add_keys=[]):
        self._survey_config = survey_config
        self._obs_table = self.__extract_from_db()
        self._field_dic = self.__init_field_dic()

    def __init_field_dic(self):
        field_list = self.obs_table['fieldID'].unique()
        dic={}
        for f in field_list:
            idx = nbf.find_first(f, self.obs_table['fieldID'].values)
            dic[f]={'ra': self.obs_table['fieldRA'][idx],
                    'dec': self.obs_table['fieldDec'][idx]}
        return dic

    @property
    def survey_config(self):
        return self._survey_config

    @property
    def band_dic(self):
        return self.survey_config['band_dic']
    @property
    def obs_table(self):
        return self._obs_table

    @property
    def field_size(self):
        """Get field size (ra,dec) in radians"""
        return np.radians(self._survey_config['ra_size']), np.radians(self._survey_config['dec_size'])

    @property
    def gain(self):
        """Get CCD gain in e-/ADU"""
        return self._survey_config['gain']

    @property
    def zp(self):
        """Get zero point"""
        if 'zp' in self._survey_config:
            return self._survey_config['zp']
        else:
            return 'zp_in_obs'

    @property
    def mintime(self):
        """Get observations mintime"""
        return self.obs_table['expMJD'].min()

    @property
    def maxtime(self):
        """Get observations maxtime"""
        return self.obs_table['expMJD'].max()

    def __extract_from_db(self):
        """Extract the observations table from SQL data base.

        Returns
        -------
        astropy.Table
            The observations table.

        """
        con = sqlite3.connect(self._survey_config['survey_file'])

        keys = ['expMJD',
                'filter',
                'fieldID',
                'fieldRA',
                'fieldDec',
                'fiveSigmaDepth']

        if 'add_data' in self.survey_config:
            add_k = (k for k in self.survey_config['add_data'] if k not in keys)
            keys+=add_k

        where = ''
        if 'survey_cut' in self.survey_config:
            cut_dic = self.survey_config['survey_cut']
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
            query +=  k+','
        query=query[:-1]
        query+= ' FROM Summary' + where + ';'
        obs_dic = pd.read_sql_query(query, con)
        return obs_dic

    def epochs_selection(self, SN):
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

        ModelMinT_obsfrm = SN.sim_model.mintime() * (1 + SN.z)
        ModelMaxT_obsfrm = SN.sim_model.maxtime() * (1 + SN.z)
        ra, dec = SN.coord

        # time selection
        # use to avoid errors
        #epochs_selec *= (self._obs_table['fiveSigmaDepth'] > 0)
        epochs_selec, selec_fields_ID = nbf.time_and_error_comp(self.obs_table['expMJD'].values,
                                                SN.sim_t0,
                                                ModelMaxT_obsfrm,
                                                ModelMinT_obsfrm,
                                                self.obs_table['fiveSigmaDepth'].values,
                                                self.obs_table['fieldID'].values)

        ra_fields = np.array(list(map(lambda x: x['ra'], map(self._field_dic.get, selec_fields_ID))))
        dec_fields = np.array(list(map(lambda x: x['dec'], map(self._field_dic.get, selec_fields_ID))))

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

        if self.zp != 'zp_in_obs':
            zp = [self.zp] * np.sum(epochs_selec)
        elif isinstance(zp, (int, float)):
            zp = self.obs_table['zp'][epochs_selec]
        else:
            raise ValueError("zp is not define")

        # Convert maglim to flux noise (ADU)
        skynoise = pw(10., 0.4 * (self.zp - mlim5)) / 5

        # Create obs table
        obs = Table({'time': self._obs_table['expMJD'][epochs_selec],
                     'band': band,
                     'gain': [self.gain] * np.sum(epochs_selec),
                     'skynoise': skynoise,
                     'zp': zp,
                     'zpsys': ['ab'] * np.sum(epochs_selec),
                     'fieldID': self._obs_table['fieldID'][epochs_selec]})
        if 'add_data' in self.survey_config:
            for k in self.survey_config['add_data']:
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
    _host_table : astropy.Table
        Description of attribute `host_list`.
    _max_dz : float
        The maximum redshift gap between 2 host.
    _z_range : list(float)
        A copy of input z_range.
    _host_file
        A copy of input host_file.

    Methods
    -------
    __read_host_file()
        Extract host from host file.
    host_in_range(host, z_range):
        Give the hosts in the good redshift range.
    random_host(n, z_range, rand_seed)
        Random choice of host in a redshift range.

    """

    def __init__(self, host_file, z_range=None):
        self._z_range = z_range
        self._host_file = host_file
        self._host_table = self.__read_host_file()
        self._max_dz = None

    @property
    def max_dz(self):
        """Get the maximum redshift gap"""
        if self._max_dz is None:
            redshift_copy = np.sort(np.copy(self.host_table['redshift']))
            diff = redshift_copy[1:] - redshift_copy[:-1]
            self._max_dz = np.max(diff)
        return self._max_dz

    @property
    def host_table(self):
        """Get astropy Table of host"""
        return self._host_table

    def __read_host_file(self):
        """Extract host from host file.

        Returns
        -------
        astropy.Table
            astropy Table containing host.

        """
        with fits.open(self._host_file) as hostf:
            host_list = hostf[1].data[:]
        host_list['ra'] = host_list['ra'] + 2 * np.pi * (host_list['ra'] < 0)
        if self._z_range is not None:
            return Table(self.host_in_range(host_list, self._z_range))
        else:
            return Table(host_list)

    @staticmethod
    def host_in_range(host, z_range):
        """Give the hosts in the good redshift range.

        Parameters
        ----------
        host : astropy.Table
            astropy Table of host.
        z_range : type
            The selection redshift range.

        Returns
        -------
        astropy.Table
            astropy Table containing host in the redshift range.

        """
        selec = host['redshift'] > z_range[0]
        selec *= host['redshift'] < z_range[1]
        return host[selec]

    def random_host(self, n, z_range, random_seed):
        """Random choice of host in a redshift range.

        Parameters
        ----------
        n : int
            Number of host to choice.
        z_range : list(float)
            The redshift range zmin, zmax.
        random_seed : int
            The random seed for the random generator.

        Returns
        -------
        astropy.Table
            astropy Table containing the randomly selected host.

        """
        if z_range[0] < self._z_range[0] or z_range[1] > self._z_range[1]:
            raise ValueError(f'z_range must be between {self._z_range[0]} and {self._z_range[1]}')
        elif z_range[0] > z_range[1]:
            raise ValueError(f'z_range[0] must be < to z_range[1]')
        host_available = self.host_in_range(self.host_table, z_range)
        host_choice = np.random.default_rng(random_seed).choice(
            host_available, size=n, replace=False)
        if len(host_choice) < n:
            raise RuntimeError('Not enough host in the shell')
        return host_choice


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
    def __init__(self,sim_lc,header):
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
        return np.array([lc.meta[key] for lc in self.sim_lc])
