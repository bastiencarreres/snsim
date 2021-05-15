"""This module contains the class which are used in the simulation"""

import sqlite3
import numpy as np
import sncosmo as snc
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from numpy import power as pw
from . import utils as ut
from .constants import C_LIGHT_KMS
from . import scatter as sct

class SN:
    """This class represent SN object.

    Parameters
    ----------
    sn_par : dict
        Contains intrinsic SN parameters generate by SNGen.
    sim_model : sncosmo.Model
        The sncosmo model used to generate the SN ligthcurve.
    model_par : dict
        Contains general model parameters and sncsomo parameters.

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
            - mb
            - x0
            - x1
            - c
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
                    test *= (self.epochs['filter'] == cut[3])
                if np.sum(test) < int(cut[0]):
                    return False
            return True

    def gen_flux(self):
        """Generate the SN lightcurve.

        Returns
        -------
        None

        Notes
        -----
        Set the sim_lc attribute as an astropy Table
        """

        params = {**{'z': self.z, 't0': self.sim_t0}, **self._model_par['sncosmo']}
        self._sim_lc = snc.realize_lcs(self.epochs, self.sim_model, [params], scatter=False)[0]
        rs = self._model_par['noise_rand_seed']
        self._sim_lc['flux'] = np.random.default_rng(rs).normal(
            loc=self.sim_lc['flux'], scale=self.sim_lc['fluxerr'])

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


class SNGen:
    """This class set up the random part of the SN generator.

    Parameters
    ----------
    sim_par : dict
        All the parameters needed to generate the SN.
    host : dict
        SnHost object containing information about SN host.

    Attributes
    ----------
    _sim_par : dict
        A copy of the input sim_par dict.
    sim_model : sncosmo.Model
        The base model to generate SN.
    _model_keys : dict
        SN model global parameters names.
    host
        SnHost object, same as input.

    Methods
    -------
    __init__(sim_par, host=None)
        Initialise the SNGen object.
    __call__(n_sn,z_range,rand_seed)
        Simulate a given number of sn in a given redshift range using the
        given random seed.
    __init_model_keys()
        Init the SN model parameters names.
    __init_sim_model()
        Configure the sncosmo Model
    """

    def __init__(self, sim_par, host=None):
        self._sim_par = sim_par
        self.sim_model = self.__init_sim_model()
        self._model_keys = ['M0']
        self._model_keys += self.__init_model_keys()
        self.host = host

    @property
    def snc_model_par(self):
        """Get sncosmo model parameters"""
        return self._sim_par['snc_model_par']

    @property
    def sn_model_par(self):
        """Get sncosmo configuration parameters"""
        return self._sim_par['sn_model_par']

    @property
    def cosmo(self):
        """Get cosmo use_defaults parameters"""
        return self._sim_par['cosmo']

    @property
    def cmb(self):
        """Get cmb used parameters"""
        return self._sim_par['cmb']

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
        model = ut.init_sn_model(self.snc_model_par['model_name'],
                                 self.snc_model_par['model_dir'])

        if 'smear_mod' in self.snc_model_par:
            smear_mod = self.snc_model_par['smear_mod']
            model = sct.init_sn_smear_model(model, smear_mod)
        return model

    def __init_model_keys(self):
        """Initialise the model keys depends on the SN simulation model.

        Returns
        -------
        list
            A dict containing all the usefull keys of the SN model.
        """
        model_name = self.snc_model_par['model_name']
        if model_name in ('salt2', 'salt3'):
            model_keys = ['alpha', 'beta']
        return model_keys

    def __call__(self, n_sn, z_range, rand_seed):
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
        rand_seeds = np.random.default_rng(rand_seed).integers(low=1000, high=100000, size=7)
        t0 = self.gen_peak_time(n_sn, rand_seeds[0])
        mag_smear = self.gen_coh_scatter(n_sn, rand_seeds[4])
        noise_rand_seed = self.gen_noise_rand_seed(n_sn, rand_seeds[5])
        model_par_sncosmo = self.gen_sncosmo_param(n_sn, rand_seeds[6:8])
        if self.host is not None:
            host = self.host.random_host(n_sn, z_range, rand_seeds[1])
            ra = host['ra']
            dec = host['dec']
            zcos = host['redshift']
            vpec = host['vp_sight']
        else:
            ra, dec = self.gen_coord(n_sn, rand_seeds[1])
            zcos = self.gen_zcos(n_sn, z_range, rand_seeds[2])
            vpec = self.gen_vpec(n_sn, rand_seeds[3])

        sn_par = [{'zcos': z,
                   'como_dist': FlatLambdaCDM(**self.cosmo).comoving_distance(z).value,
                   'z2cmb': ut.compute_z2cmb(r, d, self.cmb),
                   'sim_t0': t,
                   'ra': r,
                   'dec': d,
                   'vpec': v,
                   'mag_smear': ms
                   } for z, t, r, d, v, ms in zip(zcos, t0, ra, dec, vpec, mag_smear)]

        model_default = {}
        for k in self._model_keys:
            model_default[k] = self.sn_model_par[k]

        model_par_list = [{**model_default, 'sncosmo': mpsn, 'noise_rand_seed': rs}
                          for mpsn, rs in zip(model_par_sncosmo, noise_rand_seed)]
        SN_list = [SN(snp, self.sim_model, mp) for snp, mp in zip(sn_par, model_par_list)]
        return SN_list

    def gen_peak_time(self, n, rand_seed):
        """Generate uniformly n peak time in the survey time range.

        Parameters
        ----------
        n : int
            Number of time to generate.
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        numpy.ndarray(float)
            A numpy array which contains generated peak time.

        """
        t0 = np.random.default_rng(rand_seed).uniform(*self.sn_model_par['time_range'], size=n)
        return t0

    def gen_coord(self, n, rand_seed):
        """Generate n coords (ra,dec) uniformly on the sky sphere.

        Parameters
        ----------
        n : int
            Number of coord to generate.
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing generated coordinates.

        """
        coord_seed = np.random.default_rng(rand_seed).integers(low=1000, high=100000, size=2)
        ra = np.random.default_rng(coord_seed[0]).uniform(
            low=0, high=2 * np.pi, size=n)
        dec_uni = np.random.default_rng(coord_seed[1]).random(size=n)
        dec = np.arcsin(2 * dec_uni - 1)
        return ra, dec

    def gen_zcos(self, n, z_range, rand_seed):
        """Generate n cosmological redshift in a range.

        Parameters
        ----------
        n : int
            Number of redshift to generate.
        z_range : list(float)
            The redshift range zmin zmax.
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        numpy.ndarray(float)
            A numpy array which contains generated cosmological redshift.

        TODO: Generation is uniform, in small shell not really a problem, maybe
        fix this in general
        """
        zcos = np.random.default_rng(rand_seed).uniform(
            low=z_range[0], high=z_range[1], size=n)
        return zcos

    def gen_sncosmo_param(self, n, rand_seed):
        """Generate model dependant parameters.

        Parameters
        ----------
        n : int
            Number of parameters to generate.
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        dict
            One dictionnary containing 'parameters names': numpy.ndaray(float).

        """
        snc_seeds = np.random.default_rng(rand_seed).integers(low=1000, high=100000, size=2)
        model_name = self.snc_model_par['model_name']
        if model_name in ('salt2', 'salt3'):
            sim_x1, sim_c = self.gen_salt_par(n, snc_seeds[0])
            model_par_sncosmo = [{'x1': x1, 'c': c} for x1, c in zip(sim_x1, sim_c)]

        if 'G10_' in self.sim_model.effect_names:
            seeds = np.random.default_rng(snc_seeds[1]).integers(low=1000, high=100000, size=n)
            for par, s in zip(model_par_sncosmo, seeds):
                par['G10_RndS'] = s

        elif 'C11_' in self.sim_model.effect_names:
            seeds = np.random.default_rng(snc_seeds[1]).integers(low=1000, high=100000, size=n)
            for par, s in zip(model_par_sncosmo, seeds):
                par['C11_RndS'] = s

        return model_par_sncosmo

    def gen_salt_par(self, n, rand_seed):
        """Generate n SALT parameters.

        Parameters
        ----------
        n : int
            Number of parameters to generate.
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing SALT2 x1 and c generated parameters.

        """
        x1_seed, c_seed = np.random.default_rng(rand_seed).integers(low=1000, high=100000, size=2)
        sim_x1 = np.random.default_rng(x1_seed).normal(
            loc=self.sn_model_par['x1_distrib'][0],
            scale=self.sn_model_par['x1_distrib'][1],
            size=n)
        sim_c = np.random.default_rng(c_seed).normal(
            loc=self.sn_model_par['c_distrib'][0], scale=self.sn_model_par['c_distrib'][1], size=n)
        return sim_x1, sim_c

    def gen_vpec(self, n, rand_seed):
        """Generate n peculiar velocities.

        Parameters
        ----------
        n : int
            Number of vpec to generate.
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing vpec (km/s) generated.

        """
        vpec = np.random.default_rng(rand_seed).normal(
            loc=self.sn_model_par['vpec_distrib'][0],
            scale=self.sn_model_par['vpec_distrib'][1],
            size=n)
        return vpec

    def gen_coh_scatter(self, n, rand_seed):
        """Generate n coherent mag smear term.

        Parameters
        ----------
        n : int
            Number of mag smear terms to generate.
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing smear terms generated.

        """
        ''' Generate coherent intrinsic scattering '''
        mag_smear = np.random.default_rng(rand_seed).normal(
            loc=0, scale=self.sn_model_par['mag_smear'], size=n)
        return mag_smear

    def gen_noise_rand_seed(self, n, rand_seed):
        """Generate n seeds for later sn noise simulation.

        Parameters
        ----------
        n : int
            Number of noise seeds to generate.
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        type
            Description of returned object.

        """
        return np.random.default_rng(rand_seed).integers(low=1000, high=100000, size=n)


class ObsTable:
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

    def __init__(self, db_file, survey_prop, band_dic=None, db_cut=None, add_keys=[]):
        self._survey_prop = survey_prop
        self._db_cut = db_cut
        self._db_file = db_file
        self._band_dic = band_dic
        self._add_keys = add_keys
        self.obs_table = self.__extract_from_db()

    @property
    def field_size(self):
        """Get field size (ra,dec) in radians"""
        return self._survey_prop['ra_size'], self._survey_prop['dec_size']

    @property
    def gain(self):
        """Get CCD gain in e-/ADU"""
        return self._survey_prop['gain']

    @property
    def zp(self):
        """Get zero point"""
        if 'zp' in self._survey_prop:
            return self._survey_prop['zp']
        else:
            return 'zp_in_obs'

    @property
    def mintime(self):
        """Get observations mintime"""
        return np.min(self.obs_table['expMJD'])

    @property
    def maxtime(self):
        """Get observations maxtime"""
        return np.max(self.obs_table['expMJD'])

    def __extract_from_db(self):
        """Extract the observations table from SQL data base.

        Returns
        -------
        astropy.Table
            The observations table.

        """
        dbf = sqlite3.connect(self._db_file)

        keys = ['expMJD',
                'filter',
                'fieldRA',
                'fieldDec',
                'fiveSigmaDepth'] + self._add_keys

        where = ''
        if self._db_cut is not None:
            where = " WHERE "
            for cut_var in self._db_cut:
                where += "("
                for cut in self._db_cut[cut_var]:
                    cut_str = f"{cut}"
                    where += f"{cut_var}{cut_str} OR "
                where = where[:-4]
                where += ") AND "
            where = where[:-5]
        obs_dic = {}
        for k in keys:
            query = 'SELECT ' + k + ' FROM Summary' + where + ';'
            values = dbf.execute(query)
            obs_dic[k] = np.array([a[0] for a in values])
        return Table(obs_dic)

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
        epochs_selec = (self.obs_table['expMJD'] - SN.sim_t0 > ModelMinT_obsfrm) * \
            (self.obs_table['expMJD'] - SN.sim_t0 < ModelMaxT_obsfrm)
        # use to avoid 1e43 errors
        epochs_selec *= (self.obs_table['fiveSigmaDepth'] > 0)
        # Find the index of the field that pass time cut
        epochs_selec_idx = np.where(epochs_selec)
        # Compute the coord of the SN in the rest frame of each field
        ra_size, dec_size = self.field_size
        ra_field_frame, dec_field_frame = ut.change_sph_frame(
            ra, dec, self.obs_table['fieldRA'][epochs_selec], self.obs_table['fieldDec'][epochs_selec])
        epochs_selec[epochs_selec_idx] *= abs(ra_field_frame) < ra_size / 2  # ra selection
        epochs_selec[epochs_selec_idx] *= abs(dec_field_frame) < dec_size / 2  # dec selection
        if np.sum(epochs_selec) == 0:
            return None
        return self._make_obs_table(epochs_selec)

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
        if self._band_dic is not None:
            band = np.array(list(map(self._band_dic.get, band)))

        if self.zp != 'zp_in_obs':
            zp = [self.zp] * np.sum(epochs_selec)
        elif isinstance(zp, (int, float)):
            zp = self.obs_table['zp'][epochs_selec]
        else:
            raise ValueError("zp is not define")

        # Convert maglim to flux noise (ADU)
        skynoise = pw(10., 0.4 * (self.zp - mlim5)) / 5

        # Create obs table
        obs = Table({'time': self.obs_table['expMJD'][epochs_selec],
                     'band': band,
                     'gain': [self.gain] * np.sum(epochs_selec),
                     'skynoise': skynoise,
                     'zp': zp,
                     'zpsys': ['ab'] * np.sum(epochs_selec)})

        for k in self._add_keys:
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
