"""This module contain generators class."""
import abc
import numpy as np
import pandas as pd
import geopandas as gpd
from inspect import getsource
from .constants import C_LIGHT_KMS
from . import utils as ut
from . import nb_fun as nbf
from . import dust_utils as dst_ut
from . import scatter as sct
from . import salt_utils as salt_ut
from . import astrobj as astr
from . import constants as cst


__GEN_DIC__ = {'snia_gen': 'SNIaGen',
               'timeseries_gen':'TimeSeriesGen',
               'snii_gen': 'SNIIGen',
               'sniipl_gen': 'SNIIplGen',
               'sniib_gen': 'SNIIbGen',
               'sniin_gen': 'SNIInGen',
               'snib/c_gen':'SNIbcGen',
               'snic_gen': 'SNIcGen',
               'snib_gen': 'SNIbGen',
               'snic-bl_gen': 'SNIc_BLGen'}
               
               
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
    dipole : dict, opt
        The alpha dipole parameters.

      | dipole
      | ├── coord  list(ra, dec) dipole vector coordinates in ra, dec
      | ├── A  parameter of the A + B * cos(theta) dipole
      | └── B  B parameter of the A + B * cos(theta) dipole
    """

    # General attributes
    _available_models = []  # Flux models
    _available_rates = []   # Rate models

    def __init__(self, params, cmb, cosmology, time_range, z_range=None, peak_out_trange=False,
                 vpec_dist=None, host=None, mw_dust=None, dipole=None, geometry=None):

        if vpec_dist is not None and host is not None:
            raise ValueError("You can't set vpec_dist and host at the same time")

        # -- Mandatory parameters
        self._params = params
        self._cmb = cmb
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

        self._mw_dust = mw_dust
        self._dipole = dipole
        self._geometry = geometry

        self.rate, self._rate_expr = self._init_rate()

        # -- Init sncosmo model
        self.sim_model = self._init_sim_model()
        self._init_dust()

        # -- Init general_par
        self._general_par = {}
        self._init_general_par()

        # -- Init redshift distribution
        self._z_dist, self._z_time_rate = self._compute_zcdf()

        # -- Get the astrobj class
        self._astrobj_class = getattr(astr, self._object_type)

        if peak_out_trange:
            t0 = self.time_range[0] - self.snc_model_time[1] * (1 + self.z_range[1])
            t1 = self.time_range[1] - self.snc_model_time[0] * (1 + self.z_range[1])
            self._time_range = (t0, t1)

    def __call__(self, n_obj=None, rand_seed=None, astrobj_par=None):
        """Launch the simulation of obj.

        Parameters
        ----------
        n_obj : int
            Number of obj to simulate.
        rand_seed : int
            The random seed of the simulation.
        astrobj_par : np.records
            An array that contains pre-generated parameters

        Returns
        -------
        list(AstrObj)
            A list containing Astro Object.
        """
        if rand_seed is None:
            rand_seed = np.random.integer(1e3, 1e6)

        # -- Initialise 4 seeds for differents generation calls
        seeds = np.random.default_rng(rand_seed).integers(1e3, 1e6, size=4)

        if astrobj_par is not None:
            n_obj = len(astrobj_par)
        elif n_obj is not None:
            astrobj_par = self.gen_astrobj_par(n_obj, seed=seeds[0])
        else:
            raise ValueError('n_obj and astrobj_par cannot be None at the same time')

        # -- Add astrobj par sepecific to the obj generated
        self._update_astrobj_par(n_obj, astrobj_par, seed=seeds[1])

        # -- Add sncosmo par specific to the generated obj
        snc_par = self.gen_snc_par(n_obj, astrobj_par, seed=seeds[2])

        # -- randomly chose the number of object for each model
        rand_gen = np.random.default_rng(seeds[3])
        random_models = rand_gen.choice(list(self.sim_model.keys()), n_obj)

        # -- Check if there is dust
        if 'mw_' in self.sim_model[0].effect_names:
            dust_par = self._compute_dust_par(astrobj_par['ra'], astrobj_par['dec'])
        else:
            dust_par = [{}] * len(astrobj_par['ra'])
        
        if snc_par is not None:
            par_list = ({**{'ID': astrobj_par.index.values[i]},
                            **{k: astrobj_par[k][i+astrobj_par.index.values[0]] for k in astrobj_par},
                            **{'sncosmo': {**sncp, **dstp}}} 
                            for i, (sncp, dstp) in enumerate(zip(snc_par, dust_par)))
        else:
            par_list = ({**{'ID': astrobj_par.index.values[i]},
                            **{k: astrobj_par[k][i+astrobj_par.index.values[0]] for k in astrobj_par},
                            **{'sncosmo': { **dstp}}} 
                            for i, dstp in enumerate(dust_par))
        
        return [self._astrobj_class(snp,
                                    self.sim_model[k], 
                                    self._general_par)
                for k, snp in zip(random_models, par_list)]

    @abc.abstractmethod
    def _init_sim_model(self):
        """Abstract method that return sncosmo sim model,
        called in __init__"""
        pass

    @abc.abstractmethod
    def _update_general_par(self):
        """Abstract method to add parameters to _general_par,
        called in _init_general_par"""
        pass

    @abc.abstractmethod
    def _update_astrobj_par(self, n_obj, astrobj_par, seed=None):
        """Abstract method to add random generated parameters
        specific to the astro object used, called in __call__

        Parameters
        ----------
        astrobj_par : dict(key: np.ndarray())
            Contains basic random generated properties.
        seed : int, optional
            Random seed.
        """
        rand_gen = np.random.default_rng(seed)
        pass

    @abc.abstractmethod
    def gen_snc_par(self, n_obj, astrobj_par, seed=None):
        """Abstract method to add random generated parameters
        specific to the sncosmo model used, called in __call__

        Parameters
        ----------
        n_obj: int
            Number of parameters to generate.
        astrobj_par: dict(key: np.ndarray())
            Contains basic random generated properties.
        seed : int, optional
            Random seed.

        Return
        ------
        dict
            A dictionnary of the sncosmo parameters (not t0 or z).
        """
        rand_gen = np.random.default_rng(seed)
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

    def _init_registered_rate(self):
        """Method to give a rate to object.

        Return
        ------
        (float, float)
            Rate and rate_pw.
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
                    return eval(self._params['rate']), self._params['rate']
                # Check registered rate
                elif self._params['rate'].lower() in self._available_rates:
                    rate = self._init_registered_rate()
                    expr = ''.join(getsource(rate).partition('lambda')[1:])
                # Check for yaml bad conversion of '1e-5'
                else:
                    rate = lambda z: float(self._params['rate'])
                    expr = f"lambda z: {float(self._params['rate'])}"
            else:
                rate = lambda z: self._params['rate']
                expr = f"lambda z: {self._params['rate']}"
        # Default
        else:
            rate = lambda z: 3e-5
            expr = 'lambda z: 3e-5'
        return rate, expr.replace(',', '').replace('self.','').strip()

    def _init_general_par(self):
        """Init general parameters."""
        if self.mw_dust is not None:
            self._general_par['mw_dust'] = {'model' : self.mw_dust['model'], 
                                            'rv': self.mw_dust['rv']}
        for model in self.sim_model.values():
            if not hasattr(model, 'bandfluxcov'):
                raise ValueError('This sncosmo model has no flux covariance available')
                
        if 'mod_fcov' in self._params:
            self._general_par['mod_fcov'] = self._params['mod_fcov']
        else:
            self._general_par['mod_fcov'] = False
        self._update_general_par()

    def _init_dust(self):
        """Initialise dust."""
        if self.mw_dust is not None:
            if 'rv' not in self.mw_dust:
                self.mw_dust['rv'] = 3.1
            for model in self.sim_model.values():
                dst_ut.init_mw_dust(model, self.mw_dust)

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

    def gen_astrobj_par(self, n_obj, seed=None, min_max_t=False):
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
            * sim_t0 : obj peak
            * zcos : cosmological redshift
            * ra : Right Ascension
            * dec : Declinaison
            * vpec : peculiar velocity
            * como_dist : comoving distance
            * z2cmb : CMB dipole redshift contribution
            * mw_ebv, opt : Milky way dust extinction
            * dip_dM, opt : Dipole magnitude modification
        """
        # -- Generate seeds for random calls
        seeds = np.random.default_rng(seed).integers(1e3, 1e6, size=4)

        # -- Generate peak time
        t0 = self.gen_peak_time(n_obj, seed=seeds[0])

        # -- Generate cosmological redshifts
        if self.host is None:
            zcos = self.gen_zcos(n_obj, seed=seeds[1])
        else:
            host = self.host.random_choice(n_obj, seed=seeds[1], rate=self.rate) # change self random choiche depend on type 
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

        astrobj_par = {'zcos': zcos,
                       'como_dist': self.cosmology.comoving_distance(zcos).value,
                       'z2cmb': ut.compute_z2cmb(ra, dec, self.cmb),
                       'sim_t0': t0,
                       'ra': ra,
                       'dec': dec,
                       'vpec': vpec}

        if min_max_t:
            _1_zobs_ = (1 + astrobj_par['zcos']) 
            _1_zobs_ *= (1 + astrobj_par['z2cmb']) 
            _1_zobs_ *= (1 + astrobj_par['vpec'] / C_LIGHT_KMS)    
            astrobj_par['min_t'] = astrobj_par['sim_t0'] + self.snc_model_time[0] * _1_zobs_
            astrobj_par['max_t'] = astrobj_par['sim_t0'] + self.snc_model_time[1] * _1_zobs_
            astrobj_par['1_zobs'] = _1_zobs_

        if self.dipole is not None:
            astrobj_par['dip_dM'] = self._compute_dipole(ra, dec)

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

    def _compute_dipole(self, ra, dec):
        """Compute dipole."""
        cart_vec = nbf.radec_to_cart(self.dipole['coord'][0],
                                     self.dipole['coord'][1])

        sn_vec = nbf.radec_to_cart_2d(ra, dec)
        delta_M = self.dipole['A'] + self.dipole['B'] * sn_vec @ cart_vec
        return delta_M

    def __str__(self):
        """Print config."""
        str = ''
        
        if 'model_dir' in self._params['model_config']:
            model_dir = self._params['model_config']['model_dir']
            model_dir_str = f" from {model_dir}"
        else:
            model_dir = None
            model_dir_str = " from sncosmo"

        str += 'OBJECT TYPE : ' + self._object_type + '\n'
        str += f"SIM MODEL : {self._params['model_config']['model_name']}" + model_dir_str + '\n\n'

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
    def dipole(self):
        """Get alpha dipole parameters."""
        return self._dipole

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
    dipole : dict, opt
        The alpha dipole parameters.

      | dipole
      | ├── coord  list(ra, dec) dipole vector coordinates in ra, dec
      | ├── A  parameter of the A + B * cos(theta) dipole
      | └── B  B parameter of the A + B * cos(theta) dipole
    """
    _object_type = 'SNIa'
    _available_models = ['salt2', 'salt3']
    _available_rates = ['ptf19', 'ztf20', 'ptf19_pw']
    # M0 SNIA from JLA paper (https://arxiv.org/abs/1401.4064)
    SNIA_M0 = {
               'jla': -19.05
               }

    def _init_registered_rate(self):
        """SNIa rates registry."""
        if self._params['rate'].lower() == 'ptf19':
            # Rate from https://arxiv.org/abs/1903.08580
            return lambda z: 2.43e-5 * (self.cosmology.h / 0.70)**3
        elif self._params['rate'].lower() == 'ztf20':
            # Rate from https://arxiv.org/abs/2009.01242
            return lambda z: 2.35e-5 * (self.cosmology.h / 0.70)**3
        elif self._params['rate'].lower() == 'ptf19_pw':
            # Rate from https://arxiv.org/abs/1903.08580
            rate = 2.27e-5 * (self.cosmology.h/0.70)**3
            return lambda z: rate * (1 + z)**1.7
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")


    def _init_M0(self):
        """Initialise absolute magnitude."""
        if isinstance(self._params['M0'], (float, np.floating, int, np.integer)):
            return self._params['M0']

        elif self._params['M0'].lower() in self.SNIA_M0:
            return ut.scale_M0_cosmology(self.cosmology.h, 
                                         self.SNIA_M0[self._params['M0'].lower()], 
                                         cst.h_article[self._params['M0'].lower()])
        else: 
            raise ValueError(f"{self._params['M0']} is not available! Available M0 are {self.SNIA_M0.keys()}")  

    def _init_sim_model(self):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.

        """
        if self._params['model_config']['model_name'].lower() not in self._available_models:
            raise ValueError(f"Model {self._params['model_config']['model_name']} not available! Avaliable Models are {self._available_models}")

        model_dir = None
        if 'model_dir' in self._params['model_config']:
            model_dir = self._params['model_config']['model_dir']

        model = ut.init_sn_model(self._params['model_config']['model_name'],
                                 model_dir)

        if 'sct_model' in self._params:
            sct.init_sn_sct_model(model, self._params['sct_model'])
        return {0: model}

    def _update_general_par(self):
        """Initialise the general parameters, depends on the SN simulation model.

        Returns
        -------
        list
            A dict containing all the usefull keys of the SN model.
        """
        model_name = self._params['model_config']['model_name']
        if model_name[:5] in ('salt2', 'salt3'):
            model_keys = ['alpha', 'beta']

        self._general_par['M0'] = self._init_M0()
        self._general_par['sigM'] = self._params['sigM']

        for k in model_keys:
            self._general_par[k] = self._params['model_config'][k]
        
        return

    def _update_astrobj_par(self, n_obj, astrobj_par, seed=None):
        # -- Generate mag scattering
        if isinstance(self._params['sigM'], str):
            if self._params['sigM'].lower()== 'bs20':
                beta_sn, Rv, E_dust, c_int = sct.gen_BS20_scatter(n_obj, seed=seed)
                astrobj_par['beta_sn'] = beta_sn
                astrobj_par['c_int'] = c_int
                astrobj_par['Rv_BS20']= Rv
                astrobj_par['E_dust'] = E_dust
                astrobj_par['mag_sct'] =  c_int * beta_sn + (Rv+1) * E_dust

        else:
            astrobj_par['mag_sct'] = self.gen_coh_scatter(n_obj, seed=seed)

    def _add_print(self):
        str = ''
        if 'sct_model' in self._params:
            str += ("\nUse intrinsic scattering model : "
                    f"{self._params['sct_model']}")
        return str

    def _update_header(self, header):
        model_name = self._params['model_config']['model_name']
        header['M0_band'] = 'bessell_b'
        if model_name.lower()[:4] == 'salt':
            if isinstance(self._params['model_config']['dist_x1'], str):
                header['dist_x1'] = self._params['model_config']['dist_x1']
            else:
                header['mean_x1'] = self._params['model_config']['dist_x1'][0]
                if len(self._params['model_config']['dist_x1']) == 3:
                    header['dist_x1'] = 'asym_gauss'
                    header['sig_x1_low'] = self._params['model_config']['dist_x1'][1]
                    header['sig_x1_hi'] = self._params['model_config']['dist_x1'][2]
                elif len(self._params['model_config']['dist_x1']) == 2:
                    header['dist_x1'] = 'gauss'
                    header['sig_x1'] = self._params['model_config']['dist_x1'][1]

            
            if isinstance(self._params['model_config']['dist_c'], str):
                if self._params['model_config']['dist_c'].lower() == 'bs20':
                    header['mean_c'] = 'BS20'
                    header['dist_c'] = 'gauss c_int + gauss Edust'
                    header['sig_c'] = 'BS20'
            
            
            else:
                if len(self._params['model_config']['dist_c']) == 3:
                    header['mean_c'] = self._params['model_config']['dist_c'][0]
                    header['dist_c'] = 'asym_gauss'
                    header['sig_c_low'] = self._params['model_config']['dist_c'][1]
                    header['sig_c_hi'] = self._params['model_config']['dist_c'][2]
                else:
                    header['mean_c'] = self._params['model_config']['dist_c'][0]
                    header['dist_c'] = 'gauss'
                    header['sig_c'] = self._params['model_config']['dist_c'][1]

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
        rand_gen = np.random.default_rng(seed)

        # -- Spectra model parameters
        model_name = self._params['model_config']['model_name']

        if model_name in ('salt2', 'salt3'):
            if self._params['model_config']['dist_x1'] in ['N21']:
                z_for_dist = astrobj_par['zcos']
            else:
                z_for_dist = None
            sim_x1, sim_c = self.gen_salt_par(n_obj, rand_gen.integers(1000, 1e6),
                                              z=z_for_dist, astrobj_par=astrobj_par)
            snc_par = [{'x1': x1, 'c': c} for x1, c in zip(sim_x1, sim_c)]

        # -- Non-coherent scattering effects
        if 'G10_' in self.sim_model[0].effect_names:
            seeds = rand_gen.integers(low=1e3, high=1e6, size=n_obj)
            for par, s in zip(snc_par, seeds):
                par['G10_RndS'] = s

        elif 'C11_' in self.sim_model[0].effect_names:
            seeds = rand_gen.integers(low=1e3, high=1e6, size=n_obj)
            for par, s in zip(snc_par, seeds):
                par['C11_RndS'] = s

        return snc_par

    def gen_salt_par(self, n_sn, seed=None, z=None, astrobj_par=None):
        """Generate n SALT parameters.

        Parameters
        ----------
        n : int
            Number of parameters to generate.
        seed : int
            Random seed.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing SALT2 x1 and c generated parameters.

        """
        seeds = np.random.default_rng(seed).integers(1e3, 1e6, size=2)

        if isinstance(self._params['model_config']['dist_x1'], str):
            if self._params['model_config']['dist_x1'].lower() == 'n21':
                sim_x1 = salt_ut.n21_x1_model(z, seed=seeds[0])
        else:
            sim_x1 = ut.asym_gauss(*self._params['model_config']['dist_x1'],
                                   seed=seeds[0],
                                   size=n_sn)

        if isinstance(self._params['model_config']['dist_c'], str):
            if self._params['model_config']['dist_c'].lower() == 'bs20':
                sim_c = astrobj_par['E_dust'] + astrobj_par['c_int']
        else:
            sim_c = ut.asym_gauss(*self._params['model_config']['dist_c'],
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
    dipole : dict, opt
        The alpha dipole parameters.

      | dipole
      | ├── coord  list(ra, dec) dipole vector coordinates in ra, dec
      | ├── A  parameter of the A + B * cos(theta) dipole
      | └── B  B parameter of the A + B * cos(theta) dipole
    
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

        if isinstance(self._params['model_config']['model_name'], str):
            if self._params['model_config']['model_name'].lower() == 'all':
                selected_models = self._available_models
            elif self._params['model_config']['model_name'].lower() == 'vinc_nocorr':
                selected_models = ut.select_Vincenzi_template(self._available_models,corr=False)
            elif self._params['model_config']['model_name'].lower() == 'vinc_corr':
                selected_models = ut.select_Vincenzi_template(self._available_models,corr=True)
            else:
                selected_models = [self._params['model_config']['model_name']]

            model= [ut.init_sn_model(m) 
                    for m in selected_models]
        else:            
            model = [ut.init_sn_model(m) 
                      for m in self._params['model_config']['model_name']]
            
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

class SNIIGen(TimeSeriesGen):

    """SNII parameters generator.

    Parameters
    ----------
    same as TimeSeriesGen
    """
    _object_type = 'SNII'
    _available_models = ut.Templatelist_fromsncosmo('snii')
    _available_rates = ['ptf19', 'ztf20', 'ptf19_pw']
    
    _sn_fraction= {
                    'ztf20': 0.776208,
                    'shivers17': 0.69673 
                   }

    def _init_registered_rate(self):
        """SNII rates registry."""
        if self._params['rate'].lower() == 'ztf20':
            # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
            rate = 1.01e-4 * self._sn_fraction['ztf20']* (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19':
            # Rate from  https://arxiv.org/abs/2010.15270
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19_pw':
            # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")



    def init_M0_for_type(self):
        raise ValueError('Default M0 for SNII not implemented yet, please provide M0')

    def gen_coh_scatter_for_type(self, n_sn, seed):
        raise ValueError('Default scatterting for SNII not implemented yet, please provide SigM')

class SNIIplGen(TimeSeriesGen):
    """SNIIPL parameters generator.

    Parameters
    ----------
    same as TimeSeriesGen
    """
    _object_type = 'SNIIpl'
    _available_models = ut.Templatelist_fromsncosmo('sniipl')
    _available_rates = ['ptf19', 'ztf20', 'ptf19_pw']

    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -15.97, 'li11_skewed': -17.51},
                    'mag_sct': {'li11_gaussian': [1.31, 1.31], 'li11_skewed': [2.01, 3.18]}
                 }

    _sn_fraction={
                    'shivers17': 0.620136,
                    'ztf20': 0.546554,
                 }

    def _init_registered_rate(self):
        """SNIIPL rates registry."""
        if self._params['rate'].lower() == 'ztf20':
            # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
            rate = 1.01e-4 * self._sn_fraction['ztf20'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19':
            # Rate from  https://arxiv.org/abs/2010.15270
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19_pw':
            # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")

    def init_M0_for_type(self):
        """Initialise absolute magnitude using default values from past literature works based on the type."""
        if self._params['M0'].lower() == 'li11_gaussian':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                             self._sn_lumfunc['M0']['li11_gaussian'],
                                             cst.h_article['li11'])

        elif self._params['M0'].lower() == 'li11_skewed':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                            self._sn_lumfunc['M0']['li11_skewed'],
                                            cst.h_article['li11'])
        else:
            raise ValueError(f"{self._params['M0']} is not available! Available M0 are {self._sn_lumfunc['M0'].keys()} ")

    def gen_coh_scatter_for_type(self,n_sn, seed):
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
            

class SNIIbGen(TimeSeriesGen):
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

    def _init_registered_rate(self):
        """SNIIb rates registry."""
        if self._params['rate'].lower() == 'ztf20':
            # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
            rate = 1.01e-4 * self._sn_fraction['ztf20'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19':
            # Rate from  https://arxiv.org/abs/2010.15270
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19_pw':
            # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")

    def init_M0_for_type(self):
        """Initialise absolute magnitude using default values from past literature works based on the type."""
        if self._params['M0'].lower() == 'li11_gaussian':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                             self._sn_lumfunc['M0']['li11_gaussian'],
                                             cst.h_article['li11'])

        elif self._params['M0'].lower() == 'li11_skewed':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                            self._sn_lumfunc['M0']['li11_skewed'],
                                            cst.h_article['li11'])
        else:
            raise ValueError(f"{self._params['M0']} is not available! Available M0 are {self._sn_lumfunc['mag_scatter'].keys()} ")

    def gen_coh_scatter_for_type(self,n_sn, seed):
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
            


class SNIInGen(TimeSeriesGen):
    """SNIIn parameters generator.

    Parameters
    ----------
    same as TimeSeriesGen
    """
    _object_type = 'SNIIn'
    _available_models = ut.Templatelist_fromsncosmo('sniin')
    _available_rates = ['ptf19', 'ztf20', 'ptf19_pw']
    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -17.90, 'li11_skewed': -19.13},
                    'mag_sct': {'li11_gaussian': [0.95, 0.95], 'li11_skewed' :[1.53, 6.83]}
                 }

    _sn_fraction={
                    'shivers17': 0.046632,
                    'ztf20': 0.102524,
                 }

    def _init_registered_rate(self):
        """SNIIn rates registry."""
        if self._params['rate'].lower() == 'ztf20':
            # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
            rate = 1.01e-4 * self._sn_fraction['ztf20'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19':
            # Rate from  https://arxiv.org/abs/2010.15270
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19_pw':
            # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")

    def init_M0_for_type(self):
        """Initialise absolute magnitude using default values from past literature works based on the type."""
        if self._params['M0'].lower() == 'li11_gaussian':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                             self._sn_lumfunc['M0']['li11_gaussian'],
                                             cst.h_article['li11'])

        elif self._params['M0'].lower() == 'li11_skewed':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                            self._sn_lumfunc['M0']['li11_skewed'],
                                            cst.h_article['li11'])
        else:
            raise ValueError(f"{self._params['M0']} is not available! Available M0 are {self._sn_lumfunc['M0'].keys()} ")

    def gen_coh_scatter_for_type(self,n_sn, seed):
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

class SNIbcGen(TimeSeriesGen):

    """SNIb/c parameters generator.

    Parameters
    ----------
    same as TimeSeriesGen
    """
    _object_type = 'SNIb/c'
    _available_models = ut.Templatelist_fromsncosmo('snib/c')
    _available_rates = ['ptf19', 'ztf20', 'ptf19_pw']
    
    _sn_fraction= {
                    'ztf20': 0.217118,
                    'shivers17': 0.19456 
                   }

    def _init_registered_rate(self):
        """SNII rates registry."""
        if self._params['rate'].lower() == 'ztf20':
            # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
            rate = 1.01e-4 * self._sn_fraction['ztf20']* (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19':
            # Rate from  https://arxiv.org/abs/2010.15270
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19_pw':
            # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")



    def init_M0_for_type(self):
        raise ValueError('Default M0 for SNII not implemented yet, please provide M0')

    def gen_coh_scatter_for_type(self, n_sn, seed):
        raise ValueError('Default scatterting for SNII not implemented yet, please provide SigM')
class SNIcGen(TimeSeriesGen):
    """SNIc class.

    Parameters
    ----------
   same as TimeSeriesGen class   """
    _object_type = 'SNIc'
    _available_models =ut.Templatelist_fromsncosmo('snic')
    _available_rates = ['ptf19', 'ztf20', 'ptf19_pw']
    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -16.75, 'li11_skewed': -17.51},
                    'mag_sct': {'li11_gaussian': [0.97, 0.97], 'li11_skewed': [1.24, 1.22]}
                 }

    _sn_fraction={
                    'shivers17': 0.075088,
                    'ztf20': 0.110357,
                 }

    def _init_registered_rate(self):
        """SNIc rates registry."""
        if self._params['rate'].lower() == 'ztf20':
            # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
            rate = 1.01e-4 * self._sn_fraction['ztf20'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19':
            # Rate from  https://arxiv.org/abs/2010.15270
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19_pw':
            # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")

    def init_M0_for_type(self):
        """Initialise absolute magnitude using default values from past literature works based on the type."""
        if self._params['M0'].lower() == 'li11_gaussian':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                             self._sn_lumfunc['M0']['li11_gaussian'],
                                             cst.h_article['li11'])

        elif self._params['M0'].lower() == 'li11_skewed':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                            self._sn_lumfunc['M0']['li11_skewed'],
                                            cst.h_article['li11'])
        else:
            raise ValueError(f"{self._params['M0']} is not available! Available M0 are {self._sn_lumfunc['M0'].keys()} ")

    def gen_coh_scatter_for_type(self,n_sn, seed):
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

class SNIbGen(TimeSeriesGen):
    """SNIb class.

    Parameters
    ----------
   same as TimeSeriesGen class   """
    _object_type = 'SNIb'
    _available_models =ut.Templatelist_fromsncosmo('snib')
    _available_rates = ['ptf19', 'ztf20', 'ptf19_pw']
    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -16.07, 'li11_skewed': -17.71},
                    'mag_sct': {'li11_gaussian': [1.34, 1.34], 'li11_skewed': [2.11, 7.15]}
                 }

    _sn_fraction={
                    'shivers17': 0.108224,
                    'ztf20': 0.052551,
                 }

    def _init_registered_rate(self):
        """SNIb rates registry."""
        if self._params['rate'].lower() == 'ztf20':
            # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
            rate = 1.01e-4 * self._sn_fraction['ztf20'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19':
            # Rate from  https://arxiv.org/abs/2010.15270
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19_pw':
            # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")

    def init_M0_for_type(self):
        """Initialise absolute magnitude using default values from past literature works based on the type."""
        if self._params['M0'].lower() == 'li11_gaussian':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                             self._sn_lumfunc['M0']['li11_gaussian'],
                                             cst.h_article['li11'])

        elif self._params['M0'].lower() == 'li11_skewed':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                            self._sn_lumfunc['M0']['li11_skewed'],
                                            cst.h_article['li11'])
        else:
            raise ValueError(f"{self._params['M0']} is not available! Available M0 are {self._sn_lumfunc['M0'].keys()} ")

    def gen_coh_scatter_for_type(self,n_sn, seed):
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


class SNIc_BLGen(TimeSeriesGen):
    """SNIc_BL class.

    Parameters
    ----------
   same as TimeSeriesGen class   """
    _object_type = 'SNIc_BL'
    _available_models =ut.Templatelist_fromsncosmo('snic-bl')
    _available_rates = ['ptf19', 'ztf20', 'ptf19_pw']
    _sn_lumfunc= {
                    'M0': {'li11_gaussian': -16.79, 'li11_skewed': -17.74},
                    'mag_sct': {'li11_gaussian': [0.95, 0.95], 'li11_skewed': [1.35, 2.06]}
                 }

    _sn_fraction={
                    'shivers17': 0.011248,
                    'ztf20': 0.05421,
                 }

    def _init_registered_rate(self):
        """SNIc_BL rates registry."""
        if self._params['rate'].lower() == 'ztf20':
            # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6 
            rate = 1.01e-4 * self._sn_fraction['ztf20'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19':
            # Rate from  https://arxiv.org/abs/2010.15270
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate
        elif self._params['rate'].lower() == 'ptf19_pw':
            # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
            rate = 9.10e-5 * self._sn_fraction['shivers17'] * (self.cosmology.h/0.70)**3
            return lambda z: rate * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)
        else:
            raise ValueError(f"{self._params['rate']} is not available! Available rate are {self._available_rates}")

    def init_M0_for_type(self):
        """Initialise absolute magnitude using default values from past literature works based on the type."""
        if self._params['M0'].lower() == 'li11_gaussian':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                             self._sn_lumfunc['M0']['li11_gaussian'],
                                             cst.h_article['li11'])

        elif self._params['M0'].lower() == 'li11_skewed':
            return ut.scale_M0_cosmology(self.cosmology.h,
                                            self._sn_lumfunc['M0']['li11_skewed'],
                                            cst.h_article['li11'])
        else:
            raise ValueError(f"{self._params['M0']} is not available! Available M0 are {self._sn_lumfunc['M0'].keys()} ")

    def gen_coh_scatter_for_type(self,n_sn, seed):
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


    
class SNIa_peculiar(TimeSeriesGen):
    """SNIa_peculiar class.

    Models form platicc challenge

    Parameters
    ----------
   same as TimeSeriesGen class   """

    _object_type = 'SNIa_peculiar'
   # _available_models = need a directory to store model
    #_available_rates = 
   # _sn_lumfunc= {
                   
                # }

    #_sn_fraction={
                    
                 #}

    def _init_registered_rate(self):
        """SNIa_peculiar rates registry."""
       

    def init_M0_for_type(self):
        """Initialise absolute magnitude using default values from past literature works based on the type."""
       

    def gen_coh_scatter_for_type(self,n_sn, seed):
        """Generate n coherent mag scattering term using default values from past literature works based on the type."""
        