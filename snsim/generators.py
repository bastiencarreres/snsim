"""This module contain generators class."""
import abc
import numpy as np
import pandas as pd
import geopandas as gpd
from inspect import getsource
from .constants import C_LIGHT_KMS, VCMB, L_CMB, B_CMB
from . import utils as ut
from . import nb_fun as nbf
from . import dust_utils as dst_ut
from . import scatter as sct
from . import salt_utils as salt_ut
from . import astrobj as astr
from . import constants as cst


__GEN_DIC__ = {
    'snia_gen': 'SNIaGen',
    'timeseries_gen':'TimeSeriesGen',
    'snii_gen': 'SNIIGen',
    'sniipl_gen': 'SNIIplGen',
    'sniib_gen': 'SNIIbGen',
    'sniin_gen': 'SNIInGen',
    'snib/c_gen':'SNIbcGen',
    'snic_gen': 'SNIcGen',
    'snib_gen': 'SNIbGen',
    'snic-bl_gen': 'SNIc_BLGen'
    }


class BaseGen(abc.ABC):
    """Abstract class for basic astrobj generator.

    Parameters
    ----------
    params : dict
        Basic generator configuration.
        | params
        | ├── General obj parameters
        | └── model_config
        |     └── General sncosmo model parameters
    cmb : dict
        The CMB dipole configuration.
        | cmb
        | ├── v_cmb
        | ├── l_cmb
        | └── b_cmb
    cosmology : astropy.cosmology
        The astropy cosmological model to use.
    vpec_dist : dict
        The parameters of the peculiar velocity distribution.
        | vpec_dist
        | ├── mean_vpec
        | └── sig_vpec
    host : class SnHost, opt
        The host class to introduce sn host.
    """

    # General attributes
    _available_models = []  # Flux models
    _available_rates = {}  # Rate models

    def __init__(self, params, cosmology, time_range, z_range=None, peak_out_trange=False,
                vpec_dist=None, host=None, mw_dust=None, cmb=None, geometry=None):

        if vpec_dist is not None and host is not None:
            raise ValueError("You can't set vpec_dist and host at the same time")

        # -- Mandatory parameters
        self._params = params    
        self._cosmology = cosmology
        self._time_range = time_range

        # -- At least one mandatory
        if vpec_dist is not None and host is None:
            self._vpec_dist = vpec_dist
            self._host = None
        elif host is not None and vpec_dist is None:
            self._host = host
            self._vpec_dist = None
        else:
            raise ValueError('Set vpec_dist xor host')
        
        # -- If no host need to define a z_range
        if host is None:
            self._z_range = z_range
        else:
            self._z_range = self.host._z_range
        
        if cmb is None:
            self._cmb =  {
                'v_cmb': VCMB,
                'l_cmb': L_CMB,
                'b_cmb': B_CMB
                }
        else:
            self._cmb = cmb

        self._mw_dust = mw_dust
        self._geometry = geometry

        self.rate, self._rate_expr = self._init_rate()

        # -- Init sncosmo model
        # self.sim_model = self._init_sim_model()
        # self._init_dust()


        # -- Init redshift distribution
        self._z_dist, self._z_time_rate = self._compute_zcdf()

        # -- Get the astrobj class
        self._astrobj_class = getattr(astr, self._object_type)

        if peak_out_trange:
            t0 = self.time_range[0] - self.snc_model_time[1] * (1 + self.z_range[1])
            t1 = self.time_range[1] - self.snc_model_time[0] * (1 + self.z_range[1])
            self._time_range = (t0, t1)

    def __call__(self, n_obj=None, seed=None, basic_par=None):
        """Launch the simulation of obj.

        Parameters
        ----------
        n_obj : int
            Number of obj to simulate.
        seed : int or np.random.SeedSequence
            The random seed of the simulation.
        astrobj_par : np.records
            An array that contains pre-generated parameters

        Returns
        -------
        list(AstrObj)
            A list containing Astro Object.
        """

        # -- Initialise 4 seeds for differents generation calls
        seeds = ut.gen_rndchilds(seed, 4)

        if basic_par is not None:
            n_obj = len(basic_par)
        elif n_obj is not None:
            basic_par = self.gen_basic_par(n_obj, seed=seeds[0])
        else:
            raise ValueError('n_obj and astrobj_par cannot be None at the same time')

        # -- Add parameters specific to the generated obj
        obj_par = self.gen_par(n_obj, basic_par, seed=seeds[2])

        # -- randomly chose the number of object for each model
        # rand_gen = np.random.default_rng(seeds[3])
        # random_models = rand_gen.choice(list(self.sim_model.keys()), n_obj)

        # -- Check if there is dust
        if self.mw_dust is not None:
            dust_par = self._compute_dust_par(basic_par['ra'], basic_par['dec'])
        else:
            dust_par = [{}] * len(basic_par['ra'])
        
        return [self._astrobj_class(snp,
                                    self.sim_model[k])
                                    #self._general_par)
                for k, snp in zip(random_models, par_list)]

    def _init_registered_rate(self):
        """SNII rates registry."""
        if self._params['rate'].lower() in self._available_rates:
            return self._available_rates[self._params['rate'].lower()].format(h=self.cosmology.h)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")

    @abc.abstractmethod
    def gen_par(self, n_obj, astrobj_par, seed=None):
        """Abstract method to add random generated parameters
        specific to the astro object used, called in __call__

        Parameters
        ----------
        astrobj_par : dict(key: np.ndarray())
            Contains basic random generated properties.
        seed : int, optional
            Random seed.
        """
        pass

    def _update_header(self, header):
        """Method to add information in header,
        called in _get_header

        Parameters
        ----------
        header: dict
            dict to directly modify in the function.
        """
        pass

    def _add_print(self):
        """Method to print something in print_config."""
        pass

    def _init_rate(self):
        """Initialise rate in obj/Mpc^-3/year
        
        Returns
        -------
            lambda funtion, str
            
            The funtion and it's expression as a string
        """
        if 'rate' in self._params:
            if isinstance(self._params['rate'], type(lambda: 0)):
                rate = self._params['rate']
                expr = ''.join(getsource(self._params['rate']).partition('lambda')[1:]).replace(',', '')
            elif isinstance(self._params['rate'], str):
                # Check for lambda function in str
                if 'lambda' in self._params['rate'].lower():
                    expr = self._params['rate']
                # Check registered rate
                elif self._params['rate'].lower() in self._available_rates:
                    expr = self._init_registered_rate()
                # Check for yaml bad conversion of '1e-5'
                else:
                    expr = f"lambda z: {float(self._params['rate'])}"
            else:
                expr = f"lambda z: {self._params['rate']}"
        # Default
        else:
            expr = 'lambda z: 3e-5'
        return eval(expr), expr.strip()

    def _init_general_par(self):
        """Init general parameters."""
        if self.mw_dust is not None:
            self._general_par['mw_dust'] = self.mw_dust
        # for model in self.sim_model.values():
        #     if not hasattr(model, 'bandfluxcov'):
        #         raise ValueError('This sncosmo model has no flux covariance available')
                
        if 'mod_fcov' in self._params:
            self._general_par['mod_fcov'] = self._params['mod_fcov']
        else:
            self._general_par['mod_fcov'] = False
        #self._update_general_par()


    def gen_peak_time(self, n, seed=None):
        """Generate uniformly n peak time in the survey time range.

        Parameters
        ----------
        n : int
            Number of time to generate.
        seed : int
            Random seed.
            
        Returns
        -------
        numpy.ndarray(float)
            A numpy array which contains generated peak time.

        """
        rand_gen = np.random.default_rng(seed)

        t0 = rand_gen.uniform(*self.time_range, size=n)
        return t0

    def gen_coord(self, n, seed=None):
        """Generate n coords (ra,dec) uniformly on the sky sphere.

        Parameters
        ----------
        n : int
            Number of coord to generate.
        seed : int
            Random seed.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing generated coordinates.

        """
        rand_gen = np.random.default_rng(seed)

        if self._geometry is None:
            ra = rand_gen.uniform(low=0, high=2 * np.pi, size=n)
            dec_uni = rand_gen.random(size=n)
            dec = np.arcsin(2 * dec_uni - 1)
        else:
            # -- Init a random generator to generate multiple time
            gen_tmp = np.random.default_rng(rand_gen.integers(1e3, 1e6))
            ra, dec = [], []

            # -- Generate coord and accept if there are in the given geometry
            n_to_sim = n
            ra = []
            dec = []
            while len(ra) < n:
                ra_tmp = gen_tmp.uniform(low=0, high=2 * np.pi, size=n_to_sim)
                dec_uni_tmp = rand_gen.random(size=n_to_sim)
                dec_tmp = np.arcsin(2 * dec_uni_tmp - 1)

                multipoint = gpd.points_from_xy(ra_tmp, 
                                                dec_tmp)
                intersects = multipoint.intersects(self._geometry)
                ra.extend(ra_tmp[intersects])
                dec.extend(dec_tmp[intersects])
                n_to_sim = n - len(ra)
    
        return ra, dec

    def gen_zcos(self, n, seed=None):
        """Generate n cosmological redshift in a range.

        Parameters
        ----------
        n : int
            Number of redshift to generate.
        seed : int 
            Random seed.

        Returns
        -------
        numpy.ndarray(float)
            A numpy array which contains generated cosmological redshift.
        """
        return self._z_dist.draw(n, seed=seed)

    def gen_vpec(self, n, seed=None):
        """Generate n peculiar velocities.

        Parameters
        ----------
        n : int
            Number of vpec to generate.
        seed : int 
            Random seed.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing vpec (km/s) generated.

        """
        rand_gen = np.random.default_rng(seed)

        vpec = rand_gen.normal(
            loc=self.vpec_dist['mean_vpec'],
            scale=self.vpec_dist['sig_vpec'],
            size=n)
        return vpec

    def gen_basic_par(self, n_obj, seed=None, min_max_t=False):
        """Generate basic obj properties.

        Parameters
        ----------
        n_obj: int
            Number of obj.
        seed: int
            Random seed.

        Notes
        -----
        List of parameters:
            * t0 : obj peak
            * zcos : cosmological redshift
            * ra : Right Ascension
            * dec : Declinaison
            * vpec : peculiar velocity
            * como_dist : comoving distance
            * z2cmb : CMB dipole redshift contribution
            * mw_ebv, opt : Milky way dust extinction
        """
        # -- Generate seeds for random calls
        seeds = ut.gen_rndchilds(seed, 4)

        # -- Generate peak time
        t0 = self.gen_peak_time(n_obj, seed=seeds[0])

        # -- Generate cosmological redshifts
        if self.host is None:
            zcos = self.gen_zcos(n_obj, seed=seeds[1])
        else:
            host = self.host.random_choice(n_obj, seed=seeds[1], rate=self.rate) 
            zcos = host['zcos'].values

        # -- Generate ra, dec
        if self.host is not None:
            ra = host['ra'].values
            dec = host['dec'].values
        else:
            ra, dec = self.gen_coord(n_obj, seed=seeds[2])

        # -- Generate vpec
        if self.vpec_dist is not None:
            vpec = self.gen_vpec(n_obj, seed=seeds[3])
        elif self.host is not None:
            vpec = host['vpec'].values
        else:
            vpec = np.zeros(len(ra))

        astrobj_par = {
            'zcos': zcos,
            'como_dist': self.cosmology.comoving_distance(zcos).value,
            'z2cmb': ut.compute_z2cmb(ra, dec, self.cmb),
            't0': t0,
            'ra': ra,
            'dec': dec,
            'vpec': vpec}

        if min_max_t:
            _1_zobs_ = (1 + astrobj_par['zcos']) 
            _1_zobs_ *= (1 + astrobj_par['z2cmb']) 
            _1_zobs_ *= (1 + astrobj_par['vpec'] / C_LIGHT_KMS)    
            astrobj_par['min_t'] = astrobj_par['t0'] + self.snc_model_time[0] * _1_zobs_
            astrobj_par['max_t'] = astrobj_par['t0'] + self.snc_model_time[1] * _1_zobs_
            astrobj_par['1_zobs'] = _1_zobs_

        return pd.DataFrame(astrobj_par)

    def _compute_zcdf(self):
        """Give the time rate SN/years in redshift shell.

        Parameters
        ----------
        z : numpy.ndarray
            The redshift bins.

        Returns
        -------
        numpy.ndarray(float)
            Numpy array containing the time rate in each redshift bin.

        """
        z_min, z_max = self.z_range

        # -- Set the precision to dz = 1e-5
        dz = 1e-5

        z_shell = np.linspace(z_min, z_max, int((z_max - z_min) / dz))
        z_shell_center = 0.5 * (z_shell[1:] + z_shell[:-1])
        co_dist = self.cosmology.comoving_distance(z_shell).value
        shell_vol = 4 * np.pi / 3 * (co_dist[1:]**3 - co_dist[:-1]**3)

        # -- Compute the sn time rate in each volume shell [( SN / year )(z)]
        shell_time_rate = self.rate(z_shell_center) * shell_vol / (1 + z_shell_center)

        z_pdf = lambda x : np.interp(x, z_shell, np.append(0, shell_time_rate))

        return ut.CustomRandom(z_pdf, z_min, z_max, dx=1e-5), (z_shell, shell_time_rate)

    def _compute_dust_par(self, ra, dec):
        """Compute dust parameters.
        Parameters
        ----------
        ra : numpy.ndaray(float)
            SN Right Ascension.
        dec : numpy.ndarray(float)
            SN Declinaison.
        Returns
        -------
        list(dict)
            List of Dictionnaries that contains Rv and E(B-V) for each SN.
        """
        ebv = dst_ut.compute_ebv(ra, dec)
        mod_name = self.mw_dust['model']

        if mod_name.lower() in ['ccm89', 'od94']:
            r_v = np.ones(len(ra)) * self.mw_dust['rv']
            dust_par = [{'mw_r_v': r, 'mw_ebv': e} for r, e in zip(r_v, ebv)]
        elif mod_name.lower() in ['f99']:
            dust_par = [{'mw_ebv': e} for e in ebv]
        else:
            raise ValueError(f'{mod_name} is not implemented')
        return dust_par
    

    def __str__(self):
        """Print config."""
        str = ''
        
        if 'model_dir' in self._params:
            model_dir = self._params['model_dir']
            model_dir_str = f" from {model_dir}"
        else:
            model_dir = None
            model_dir_str = " from sncosmo"

        str += 'OBJECT TYPE : ' + self._object_type + '\n'
        str += f"SIM MODEL : {self._params['model_name']}" + model_dir_str + '\n\n'

        str += ("Peak mintime : "
                f"{self.time_range[0]:.2f} MJD\n\n"
                "Peak maxtime : "
                f"{self.time_range[1]:.2f} MJD\n\n")
        
        str += 'Redshift distribution computed'

        if self.host is not None:
            if self.host.config['distrib'] == 'random':
                str += ' using host redshift distribution\n'
            elif  self.host.config['distrib'] == 'survey_rate':
                str += ' using rate\n\n'
        else:
            str += ' using rate\n'

        
        str += self._add_print() + '\n'

        if self._general_par['mod_fcov']:
            str += "Model COV ON"
        else:
            str += "Model COV OFF"
        return str

    def _get_header(self):
        """Generate header of sim file..

        Returns
        -------
        None

        """
        header = {
                'obj_type': self._object_type,
                'rate': self._rate_expr,
                'model_name': [model.source.name 
                            for model in self.sim_model.values()]
                }

        header = {**header, **self._general_par}
    
        if self.vpec_dist is not None:
            header['m_vp'] = self.vpec_dist['mean_vpec']
            header['s_vp'] = self.vpec_dist['sig_vpec']

        self._update_header(header)
        return header

    @property
    def snc_model_time(self):
        """Get the sncosmo model mintime and maxtime."""
        maxtime = np.max([model.maxtime() for model in self.sim_model.values()])
        mintime = np.min([model.mintime() for model in self.sim_model.values()])
        return mintime, maxtime

    @property
    def host(self):
        """Get the host class."""
        return self._host

    @property
    def vpec_dist(self):
        """Get the peculiar velocity distribution parameters."""
        return self._vpec_dist

    @property
    def mw_dust(self):
        """Get the mw_dust parameters."""
        return self._mw_dust

    @property
    def cosmology(self):
        """Get astropy cosmological model."""
        return self._cosmology

    @property
    def cmb(self):
        """Get cmb used parameters."""
        return self._cmb

    @property
    def time_range(self):
        """Get time range."""
        return self._time_range

    @property
    def z_range(self):
        """Get redshift range."""
        return self._z_range
    
    @property
    def z_cdf(self):
        """Get the redshift cumulative distribution."""
        if self._z_dist is None:
            return None
        return self._z_dist.cdf


class SNIaGen(BaseGen):
    """SNIa parameters generator.

    Parameters
    ----------
    params : dict
        Basic params + SNIa specific parameters.

      | params
      | ├── M0, SNIa absolute magnitude
      | ├── sigM, SNIa coherent scattering
      | └── sct_model, opt, SNIa wavelenght dependant scattering
    cmb : dict
        The CMB dipole configuration.

      | cmb
      | ├── vcmb
      | ├── l_cmb
      | └── b_cmb
    cosmology : astropy.cosmology
        The astropy cosmological model to use.
    vpec_dist : dict
        The parameters of the peculiar velocity distribution.

      | vpec_dist
      | ├── mean_vpec
      | └── sig_vpec
    host : class SnHost, opt
        The host class to introduce sn host.
    """
    _object_type = 'SNIa'
    _available_models = ['salt2', 'salt3']
    _available_rates = {
        'ptf19': 'lambda z:  2.43e-5 * ({h}/0.70)**3', # Rate from https://arxiv.org/abs/1903.08580
        'ztf20': 'lambda z:  2.35e-5 * ({h}/0.70)**3', # Rate from https://arxiv.org/abs/2009.01242
        'ptf19_pw':  'lambda z:  2.35e-5 * ({h}/0.70)**3 * (1 + z)**1.7' # Rate from https://arxiv.org/abs/1903.08580
        }
    
    SNIA_M0 = {
            'jla': -19.05  # M0 SNIA from JLA paper (https://arxiv.org/abs/1401.4064)
            }

    def _init_M0(self):
        """Initialise absolute magnitude."""
        if isinstance(self._params['M0'], (float, np.floating, int, np.integer)):
            return self._params['M0']

        elif self._params['M0'].lower() in self.SNIA_M0:
            return ut.scale_M0_cosmology(
                self.cosmology.h, 
                self.SNIA_M0[self._params['M0'].lower()], 
                cst.h_article[self._params['M0'].lower()])
        else: 
            raise ValueError(f"{self._params['M0']} is not available! Available M0 are {self.SNIA_M0.keys()}")  


    # def _update_general_par(self):
    #     """Initialise the general parameters, depends on the SN simulation model.

    #     Returns
    #     -------
    #     list
    #         A dict containing all the usefull keys of the SN model.
    #     """
    #     model_name = self._params['model_name']
    #     if model_name[:5] in ('salt2', 'salt3'):
    #         model_keys = ['alpha', 'beta']

    #     self._general_par['M0'] = self._init_M0()
    #     self._general_par['sigM'] = self._params['sigM']

    #     for k in model_keys:
    #         self._general_par[k] = self._params['model_config'][k]
    #     return


    def _add_print(self):
        str = ''
        if 'sct_model' in self._params:
            str += ("\nUse intrinsic scattering model : "
                    f"{self._params['sct_model']}")
        return str

    def _update_header(self, header):
        model_name = self._params['model_name']
        header['M0_band'] = 'bessell_b'
        if model_name.lower()[:4] == 'salt':
            if isinstance(self._params['dist_x1'], str):
                header['dist_x1'] = self._params['dist_x1']
            else:
                header['mean_x1'] = self._params['dist_x1'][0]
                if len(self._params['dist_x1']) == 3:
                    header['dist_x1'] = 'asym_gauss'
                    header['sig_x1_low'] = self._params['dist_x1'][1]
                    header['sig_x1_hi'] = self._params['dist_x1'][2]
                elif len(self._params['dist_x1']) == 2:
                    header['dist_x1'] = 'gauss'
                    header['sig_x1'] = self._params['dist_x1'][1]

            header['mean_c'] = self._params['dist_c'][0]

            if len(self._params['dist_c']) == 3:
                header['dist_c'] = 'asym_gauss'
                header['sig_c_low'] = self._params['dist_c'][1]
                header['sig_c_hi'] = self._params['dist_c'][2]
            else:
                header['dist_c'] = 'gauss'
                header['sig_c'] = self._params['dist_c'][1]

        if 'sct_model' in self._params:
            header['sct_mod'] = self._params['sct_model']
            if self._params['sct_model'].lower() == 'g10':
                params = ['G10_L0', 'G10_F0', 'G10_F1', 'G10_dL']
                for par in params:
                    pos = np.where(np.array(self.sim_model[0].param_names) == par)[0]
                    header[par] = self.sim_model[0].parameters[pos][0]
            elif self._params['sct_model'].lower() == 'c11':
                params = ['C11_Cuu', 'C11_Sc']
                for par in params:
                    pos = np.where(np.array(self.sim_model[0].param_names) == par)[0]
                    header[par] = self.sim_model[0].parameters[pos][0]

    def gen_coh_scatter(self, n_sn, seed=None):
        """Generate n coherent mag scattering term.

        Parameters
        ----------
        n : int
            Number of mag scattering terms to generate.
        seed : int, optional
            Random seed.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing scattering terms generated.

        """
        rand_gen = np.random.default_rng(seed)

        mag_sct = rand_gen.normal(loc=0, scale=self._params['sigM'], size=n_sn)
        return mag_sct

    def gen_par(self, n_obj, astrobj_par, seed=None):
        """Generate SNIa specific parameters.

        Parameters
        ----------
        n_obj : int
            Number of parameters to generate.
        seed : int, optional
            Random seed
            .

        Returns
        -------
        dict
            One dictionnary containing 'parameters names': numpy.ndarray(float).

        """
        seeds = ut.gen_rndchilds(seed=seed, size=3)
        
        self.gen_coh_scatter(n_obj, seed=seeds[0])
        # -- Spectra model parameters
        model_name = self._params['model_name']

        if model_name in ('salt2', 'salt3'):
            sim_x1, sim_c = self.gen_salt_par(
                n_obj,
                seeds[1],
                z=astrobj_par['zcos'])
            params = {'x1': sim_x1, 'c': sim_c}

        # -- Non-coherent scattering effects
        if 'sct_model' in self._params:
            if self._params['sct_model'] == 'G10':
                params['G10_RndS'] = np.array(seeds[2].spawn(n_obj))
            elif self._params['sct_model'] == 'C11':
                params['G10_RndS'] = np.array(seeds[2].spawn(n_obj))
        return params

    def gen_salt_par(self, n_sn, seed=None, z=None):
        """Generate SALT parameters.

        Parameters
        ----------
        n_sn : int
            Number of parameters to generate.
        seed : int
            Random seed.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing SALT2 x1 and c generated parameters.

        """
        seeds = ut.gen_rndchilds(seed=seed, size=2)

        if isinstance(self._params['dist_x1'], str):
            if self._params['dist_x1'].lower() == 'n21':
                sim_x1 = salt_ut.n21_x1_model(z, seed=seeds[0])
        else:
            sim_x1 = ut.asym_gauss(*self._params['dist_x1'],
                                   seed=seeds[0],
                                   size=n_sn)

        sim_c = ut.asym_gauss(*self._params['dist_c'],
                              seed=seeds[1],
                              size=n_sn)
        return sim_x1, sim_c



class TimeSeriesGen(BaseGen):
    """TimeSeries parameters generator.

    Parameters
    ----------
    params : dict
        Basic params + TimeSeries specific parameters.

      | params
      | ├── M0, absolute magnitude
      | ├── sigM, coherent scattering
      |
    cmb : dict
        The CMB dipole configuration.

      | cmb
      | ├── vcmb
      | ├── l_cmb
      | └── b_cmb
    cosmology : astropy.cosmology
        The astropy cosmological model to use.
    vpec_dist : dict
        The parameters of the peculiar velocity distribution.

      | vpec_dist
      | ├── mean_vpec
      | └── sig_vpec
    host : class SnHost, opt
        The host class to introduce sn host.
    
    General Info about parameters:

    For Rate:
    SNCC ztf20 relative fraction of SNe subtypes from https://arxiv.org/abs/2009.01242 figure 6 + 
    ztf20 relative fraction between SNe Ic and SNe Ib from https://iopscience.iop.org/article/10.3847/1538-4357/aa5eb7/meta
    SNCC shiver17 fraction from https://arxiv.org/abs/1609.02922 Table 3

    For Luminosity Functions:
    SNCC M0 mean and scattering of luminosity function values from Vincenzi et al. 2021 Table 5 (https://arxiv.org/abs/2111.10382)
    """

    def _init_M0(self):
        """Initialise absolute magnitude."""
        if isinstance(self._params['M0'], (float, np.floating, int, np.integer)):
            return self._params['M0']

        else:
            return self.init_M0_for_type()
         
    def _init_sim_model(self):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.
        """

        if isinstance(self._params['model_name'], str):
            if self._params['model_name'].lower() == 'all':
                selected_models = self._available_models
            elif self._params['model_name'].lower() == 'vinc_nocorr':
                selected_models = ut.select_Vincenzi_template(self._available_models,corr=False)
            elif self._params['model_name'].lower() == 'vinc_corr':
                selected_models = ut.select_Vincenzi_template(self._available_models,corr=True)
            else:
                selected_models = [self._params['model_name']]

            model= [ut.init_sn_model(m) 
                    for m in selected_models]
        else:            
            model = [ut.init_sn_model(m) 
                      for m in self._params['model_name']]
            
        model = {i :m for i, m in enumerate(model)}
        
        return model


    def _update_general_par(self):
        """Initialise the general parameters, depends on the SN simulation model.

        Returns
        -------
        list
            A dict containing all the usefull keys of the SN model.
        """

        self._general_par['M0'] = self._init_M0()
        self._general_par['sigM'] = self._params['sigM']
       
        return

    def _update_astrobj_par(self, n_obj, astrobj_par, seed=None):
        # -- Generate coherent mag scattering
        astrobj_par['mag_sct'] = self.gen_coh_scatter(n_obj, seed=seed)

   
    def gen_coh_scatter(self, n_sn, seed=None):
        """Generate n coherent mag scattering term.

        Parameters
        ----------
        n : int
            Number of mag scattering terms to generate.
        seed : int, optional
            Random seed.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing scattering terms generated.

        """
        if seed is None:
            seed = np.random.random_integers(1e3, 1e6)
        rand_gen = np.random.default_rng(seed)
        
        if isinstance(self._params['sigM'], (float, np.floating, int, np.integer)):
            return rand_gen.normal(loc=0, scale=self._params['sigM'], size=n_sn)

        elif isinstance(self._params['sigM'],list):
            return ut.asym_gauss(mu=0,
                    sig_low=self._params['sigM'][0],
                    sig_high=self._params['sigM'][1], seed=seed, size=n_sn)

        else:
            return self.gen_coh_scatter_for_type(n_sn, seed)
            

    
    def gen_snc_par(self, n_obj, astrobj_par, seed=None):
        """Generate sncosmo model dependant parameters (others than redshift and t0).
        Parameters
        ----------
        n_obj : int
            Number of parameters to generate.
        seed : int, optional
            Random seed
            .
        Returns
        -------
        dict
            One dictionnary containing 'parameters names': numpy.ndarray(float).
        """
       
        return None

    def _add_print(self):
        str = ''
        return str

    def _update_header(self, header):
        header['M0_band']='bessell_r'

class CCGen(TimeSeriesGen):
    """Template for CoreColapse."""
    
    def init_M0_for_type():
        """Initialise absolute magnitude using default values from past literature works based on the type."""
        if self._params['M0'].lower() == 'li11_gaussian':
            return ut.scale_M0_cosmology(
                    self.cosmology.h,
                    self._sn_lumfunc['M0']['li11_gaussian'],
                    cst.h_article['li11']
                    )

        elif self._params['M0'].lower() == 'li11_skewed':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                            self._sn_lumfunc['M0']['li11_skewed'],
                                            cst.h_article['li11'])
        else:
            raise ValueError(f"{self._params['M0']} is not available! Available M0 are {self._sn_lumfunc['M0'].keys()} ")

    def gen_coh_scatter_for_type(n_sn, seed):
        """Generate n coherent mag scattering term using default values from past literature works based on the type."""
        if self._params['sigM'].lower() == 'li11_gaussian':
            return ut.asym_gauss(mu=0,
                    sig_low=self._sn_lumfunc['mag_sct']['li11_gaussian'][0],
                    sig_high=self._sn_lumfunc['mag_sct']['li11_gaussian'][1],
                    seed=seed, size=n_sn) 
            
        elif self._params['sigM'].lower() == 'li11_skewed':
            return ut.asym_gauss(mu=0,
                    sig_low=self._sn_lumfunc['mag_sct']['li11_skewed'][0],
                    sig_high=self._sn_lumfunc['mag_sct']['li11_skewed'][1],
                    seed=seed, size=n_sn)
        else:
            raise ValueError(f"{self._params['sigM']} is not available! Available sigM are {self._sn_lumfunc['mag_scatter'].keys()} ")
    
    
class SNIIGen(CCGen):
    """SNII parameters generator.

    Parameters
    ----------
    same as TimeSeriesGen
    """
    _object_type = 'SNII'
    _available_models = ut.Templatelist_fromsncosmo('snii')
    
    _sn_fraction = {
                    'ztf20': 0.776208,
                    'shivers17': 0.69673 
                    }
    
    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
        'ptf19' : f"lambda z: 1.01e-4 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3", 
        # Rate from  https://arxiv.org/abs/2010.15270
        'ztf20': f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        'ptf19_pw': f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3  * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6"
                        }

    def init_M0_for_type(self):
        raise ValueError('Default M0 for SNII not implemented yet, please provide M0')

    def gen_coh_scatter_for_type(self, n_sn, seed):
        raise ValueError('Default scatterting for SNII not implemented yet, please provide SigM')


class SNIIplGen(CCGen):
    """SNIIPL parameters generator.

    Parameters
    ----------
    same as TimeSeriesGen
    """
    _object_type = 'SNIIpl'
    _available_models = ut.Templatelist_fromsncosmo('sniipl')

    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -15.97, 'li11_skewed': -17.51},
                    'mag_sct': {'li11_gaussian': [1.31, 1.31], 'li11_skewed': [2.01, 3.18]}
                }

    _sn_fraction={
                    'shivers17': 0.620136,
                    'ztf20': 0.546554,
                }
    
    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
        'ptf19': f"1.01e-4 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        'ztf20': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        'ptf19_pw': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)"
        }
        

class SNIIbGen(CCGen):
    """SNIIb parameters generator.

    Parameters
    ----------
    same as TimeSeriesGen
    """
    _object_type = 'SNIIb'
    _available_models = ut.Templatelist_fromsncosmo('sniib')
    _available_rates = ['ptf19', 'ztf20', 'ptf19_pw']
    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -16.69, 'li11_skewed': -18.30},
                    'mag_sct': {'li11_gaussian': [1.38, 1.38], 'li11_skewed': [2.03, 7.40]}
                }

    _sn_fraction={
                    'shivers17': 0.10944,
                    'ztf20': 0.047652,
                }
    
    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
        'ptf19': f"1.01e-4 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        'ztf20': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        'ptf19_pw': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)"
        }
        
    def _init_registered_rate(self):
        """SNIIPL rates registry."""
        if self._params['rate'].lower() in self._available_rates:
            return self._available_rates[self._params['rate'].lower()].format(h=self.cosmology.h)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")
        

class SNIInGen(CCGen):
    """SNIIn parameters generator.

    Parameters
    ----------
    same as TimeSeriesGen
    """
    _object_type = 'SNIIn'
    _available_models = ut.Templatelist_fromsncosmo('sniin')

    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -17.90, 'li11_skewed': -19.13},
                    'mag_sct': {'li11_gaussian': [0.95, 0.95], 'li11_skewed' :[1.53, 6.83]}
                }

    _sn_fraction={
                    'shivers17': 0.046632,
                    'ztf20': 0.102524,
                }
    
    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
        'ptf19': f"1.01e-4 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        'ztf20': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        'ptf19_pw': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)"
        }
        

class SNIbcGen(CCGen):

    """SNIb/c parameters generator.

    Parameters
    ----------
    same as TimeSeriesGen
    """
    _object_type = 'SNIb/c'
    _available_models = ut.Templatelist_fromsncosmo('snib/c')    
    _sn_fraction= {
                    'ztf20': 0.217118,
                    'shivers17': 0.19456 
                }

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
        'ptf19': f"1.01e-4 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        'ztf20': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        'ptf19_pw': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)"
        }
        
    def init_M0_for_type(self):
        raise ValueError('Default M0 for SNII not implemented yet, please provide M0')

    def gen_coh_scatter_for_type(self, n_sn, seed):
        raise ValueError('Default scatterting for SNII not implemented yet, please provide SigM')
    

class SNIcGen(CCGen):
    """SNIc class.

    Parameters
    ----------
    same as TimeSeriesGen class   """
    _object_type = 'SNIc'
    _available_models =ut.Templatelist_fromsncosmo('snic')
    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -16.75, 'li11_skewed': -17.51},
                    'mag_sct': {'li11_gaussian': [0.97, 0.97], 'li11_skewed': [1.24, 1.22]}
                }

    _sn_fraction={
                    'shivers17': 0.075088,
                    'ztf20': 0.110357,
                }
    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
        'ptf19': f"1.01e-4 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        'ztf20': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        'ptf19_pw': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)"
        }
        

class SNIbGen(CCGen):
    """SNIb class.

    Parameters
    ----------
    same as TimeSeriesGen class."""
    _object_type = 'SNIb'
    _available_models =ut.Templatelist_fromsncosmo('snib')
    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -16.07, 'li11_skewed': -17.71},
                    'mag_sct': {'li11_gaussian': [1.34, 1.34], 'li11_skewed': [2.11, 7.15]}
                 }

    _sn_fraction={
                    'shivers17': 0.108224,
                    'ztf20': 0.052551,
                }
    
    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
        'ptf19': f"1.01e-4 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        'ztf20': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        'ptf19_pw': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)"
        }


class SNIc_BLGen(CCGen):
    """SNIc_BL class.

    Parameters
    ----------
    Same as TimeSeriesGen class."""
    _object_type = 'SNIc_BL'
    _available_models =ut.Templatelist_fromsncosmo('snic-bl')
    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -16.79, 'li11_skewed': -17.74},
                    'mag_sct': {'li11_gaussian': [0.95, 0.95], 'li11_skewed': [1.35, 2.06]}
                }

    _sn_fraction={
                    'shivers17': 0.011248,
                    'ztf20': 0.05421,
                }
    
    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
        'ptf19': f"1.01e-4 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        'ztf20': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        'ptf19_pw': f"9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)"
        }


class SNIa_peculiar(BaseGen):
    """SNIa_peculiar class.

    Models form platicc challenge ask Rick
    need a directory to store model

    Parameters
    ----------
   same as TimeSeriesGen class   """

    _object_type = 'SNIa_peculiar'
   # _available_models = 
    #_available_rates = 
   # _sn_lumfunc= {
                   
                # }

    #_sn_fraction={
                    
                 #}


    def _init_M0(self):
        """Initialise absolute magnitude."""
        if isinstance(self._params['M0'], (float, np.floating, int, np.integer)):
            return self._params['M0']

        else:
            return self.init_M0_for_type()
         
    def _init_sim_model(self):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.
        """

        if isinstance(self._params['model_name'], str):
            if self._params['model_name'].lower() == 'all':
                selected_models = self._available_models
            elif self._params['model_name'].lower() == 'vinc_nocorr':
                selected_models = ut.select_Vincenzi_template(self._available_models,corr=False)
            elif self._params['model_name'].lower() == 'vinc_corr':
                selected_models = ut.select_Vincenzi_template(self._available_models,corr=True)
            else:
                selected_models = [self._params['model_name']]

            model= [ut.init_sn_model(m) 
                    for m in selected_models]
        else:            
            model = [ut.init_sn_model(m) 
                      for m in self._params['model_name']]
        
        model = {i :m for i, m in enumerate(model)}
        
        return model


    def _update_general_par(self):
        """Initialise the general parameters, depends on the SN simulation model.

        Returns
        -------
        list
            A dict containing all the usefull keys of the SN model.
        """

        self._general_par['M0'] = self._init_M0()
        self._general_par['sigM'] = self._params['sigM']
       
        return

    def _update_astrobj_par(self, n_obj, astrobj_par, seed=None):
        # -- Generate coherent mag scattering
        astrobj_par['mag_sct'] = self.gen_coh_scatter(n_obj, seed=seed)

   
    def gen_coh_scatter(self, n_sn, seed=None):
        """Generate n coherent mag scattering term.

        Parameters
        ----------
        n : int
            Number of mag scattering terms to generate.
        seed : int, optional
            Random seed.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing scattering terms generated.

        """
        if seed is None:
            seed = np.random.random_integers(1e3, 1e6)
        rand_gen = np.random.default_rng(seed)
        
        if isinstance(self._params['sigM'], (float, np.floating, int, np.integer)):
            return rand_gen.normal(loc=0, scale=self._params['sigM'], size=n_sn)

        elif isinstance(self._params['sigM'],list):
            return ut.asym_gauss(mu=0,
                    sig_low=self._params['sigM'][0],
                    sig_high=self._params['sigM'][1], seed=seed, size=n_sn)

        else:
            return self.gen_coh_scatter_for_type(n_sn, seed)
            

    
    def gen_snc_par(self, n_obj, astrobj_par, seed=None):
        """Generate sncosmo model dependant parameters (others than redshift and t0).
        Parameters
        ----------
        n_obj : int
            Number of parameters to generate.
        seed : int, optional
            Random seed
            .
        Returns
        -------
        dict
            One dictionnary containing 'parameters names': numpy.ndarray(float).
        """
       
        return None

    def _add_print(self):
        str = ''
        return str

    def _update_header(self, header):
        header['M0_band']='bessell_r'


    def _init_registered_rate(self):
        """SNIa_peculiar rates registry."""
       

    def init_M0_for_type():
        """Initialise absolute magnitude using default values from past literature works based on the type."""
       

    def gen_coh_scatter_for_type(n_sn, seed):
        """Generate n coherent mag scattering term using default values from past literature works based on the type."""


class SNIax(SNIa_peculiar):
    """SNIaxclass.

    Models form platicc challenge ask Rick
    need a directory to store model

    Parameters
    ----------
   same as TimeSeriesGen class   """
   
   
   
class SNIa_91bg(SNIa_peculiar):
    """SNIa 91bg-like class.

    Models form platicc challenge ask Rick
    need a directory to store model

    Parameters
    ----------
   same as TimeSeriesGen class   """