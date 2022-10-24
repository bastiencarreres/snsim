"""This module contain generators class."""

import abc
import numpy as np
from . import utils as ut
from . import nb_fun as nbf
from . import dust_utils as dst_ut
from . import scatter as sct
from . import salt_utils as salt_ut
from . import astrobj as astr
from shapely import geometry as shp_geo
import pandas as pd


__GEN_DIC__ = {'snia_gen': 'SNIaGen'}


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
    _object_type = ''
    _available_models = []  # Flux models
    _available_rates = []   # Rate models

    def __init__(self, params, cmb, cosmology, vpec_dist=None,
                 host=None, mw_dust=None, dipole=None, survey_footprint=None):

        if vpec_dist is not None and host is not None:
            raise ValueError("You can't set vpec_dist and host at the same time")

        self._params = params
        self._cmb = cmb
        self._cosmology = cosmology
        self._vpec_dist = vpec_dist
        self._host = host
        self._mw_dust = mw_dust
        self._dipole = dipole
        self._footprint = survey_footprint

        self.rate_law = self._init_rate()

        # -- Init sncosmo model
        self.sim_model = self._init_sim_model()
        self._init_dust()

        # -- Init general_par
        self._general_par = {}
        self._init_general_par()

        self._time_range = None
        self._z_dist = None
        self._z_time_rate = None

        # -- Get the astrobj class
        self._astrobj_class = getattr(astr, self._object_type)

    def __call__(self, n_obj, rand_seed, astrobj_par=None):
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
        # -- Initialise 4 seeds for differents generation calls
        seeds = np.random.default_rng(rand_seed).integers(1e3, 1e6, size=4)

        if astrobj_par is None:
            astrobj_par = self.gen_astrobj_par(n_obj, seed=seeds[0])

        # -- Add astrobj par sepecific to the obj generated
        self._update_astrobj_par(n_obj, astrobj_par, seed=seeds[1])

        # -- Add sncosmo par specific to the generated obj
        snc_par = self.gen_snc_par(n_obj, astrobj_par, seed=seeds[2])

        # -- Check if there is dust
        if 'mw_' in self.sim_model.effect_names:
            dust_par = self._compute_dust_par(astrobj_par['ra'], astrobj_par['dec'])
        else:
            dust_par = [{}] * len(astrobj_par['ra'])

        astrobj_list = ({**{k: astrobj_par[k][i] for k in astrobj_par},
                         **{'sncosmo': {**sncp, **dstp}}} for i, (sncp, dstp) in enumerate(zip(snc_par, dust_par)))

        return [self._astrobj_class(snp, self.sim_model, self._general_par) for snp in astrobj_list]

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
        """Initialise rate in obj/Mpc^-3/year"""
        if 'rate' in self._params:
            if isinstance(self._params['rate'], str):
                # Check registered rate
                if self._params['rate'].lower() in self._available_rates:
                    return self._init_registered_rate()

                # Check for the yaml bad conversion of '1e-5'
                else:
                    rate = float(self._params['rate'])
        else:
            # Default rate
            rate = 3e-5

        if 'rate_pw' in self._params:
            rate_pw = self._params['rate_pw']
        else:
            # Default rate powerlaw
            rate_pw = 0
        return rate, rate_pw

    def _init_general_par(self):
        """Init general parameters."""
        if self.mw_dust is not None:
            self._general_par['mw_dust'] = self.mw_dust['model']
            self._general_par['mw_rv'] = self.mw_dust['rv']
        if not hasattr(self.sim_model, 'bandfluxcov'):
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
                dst_ut.init_mw_dust(self.sim_model, self.mw_dust)

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

        if self._footprint is None:
            ra = rand_gen.uniform(low=0, high=2 * np.pi, size=n)
            dec_uni = rand_gen.random(size=n)
            dec = np.arcsin(2 * dec_uni - 1)
        else:
            # -- Init a random generator to generate multiple time
            gen_tmp = np.random.default_rng(rand_gen.integers(1e3, 1e6))
            ra, dec = [], []

            # -- Generate coord and accept if there are in footprint
            while len(ra) < n:
                ra_tmp = gen_tmp.uniform(low=0, high=2 * np.pi)
                dec_uni_tmp = rand_gen.random()
                dec_tmp = np.arcsin(2 * dec_uni_tmp - 1)
                if self._footprint.contains(shp_geo.Point(ra_tmp, dec_tmp)):
                    ra.append(ra_tmp)
                    dec.append(dec_tmp)
            ra = np.array(ra)
            dec = np.array(dec)
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

    def gen_astrobj_par(self, n_obj, seed=None):
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
            hseed = rand_gen.integers(1e3, 1e6)
            host = self.host.random_choice(n_obj, seed=seeds[1], z_cdf=self.z_cdf)
            zcos = host['redshift'].values

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
            vpec = host['v_radial'].values
        else:
            vpec = np.zeros(len(ra))

        astrobj_par = {'zcos': zcos,
                       'como_dist': self.cosmology.comoving_distance(zcos).value,
                       'z2cmb': ut.compute_z2cmb(ra, dec, self.cmb),
                       'sim_t0': t0,
                       'ra': ra,
                       'dec': dec,
                       'vpec': vpec}

        if self.dipole is not None:
            astrobj_par['dip_dM'] = self._compute_dipole(ra, dec)

        return pd.DataFrame(astrobj_par)

    def rate(self, z):
        """Give the rate SNs/Mpc^3/year at redshift z.

        Parameters
        ----------
        z : float, numpy.ndarray(float)
            One of a list of cosmological redshift(s).

        Returns
        -------
        float, numpy.ndarray(float)
            One or a list of sn rate(s) corresponding to input redshift(s).

        """
        rate_z0, rpw = self.rate_law
        return rate_z0 * (1 + z)**rpw

    def compute_zcdf(self, z_range):
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
        z_min, z_max = z_range

        # -- Set the precision to dz = 1e-5
        dz = 1e-5

        z_shell = np.linspace(z_min, z_max, int((z_max - z_min) / dz))
        z_shell_center = 0.5 * (z_shell[1:] + z_shell[:-1])
        rate = self.rate(z_shell_center)  # Rate in Nsn/Mpc^3/year
        co_dist = self.cosmology.comoving_distance(z_shell).value
        shell_vol = 4 * np.pi / 3 * (co_dist[1:]**3 - co_dist[:-1]**3)

        # -- Compute the sn time rate in each volume shell [( SN / year )(z)]
        shell_time_rate = rate * shell_vol / (1 + z_shell_center)

        z_pdf = lambda x : np.interp(x, z_shell, np.append(0, shell_time_rate))

        self._z_dist = ut.CustomRandom(z_pdf, z_min, z_max)
        self._z_time_rate = z_shell, shell_time_rate

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

    def print_config(self):
        """Print config."""
        if 'model_dir' in self._params['model_config']:
            model_dir = self._params['model_config']['model_dir']
            model_dir_str = f" from {model_dir}"
        else:
            model_dir = None
            model_dir_str = " from sncosmo"

        print('OBJECT TYPE : ' + self._object_type)
        print(f"SIM MODEL : {self._params['model_config']['model_name']}" + model_dir_str)

        self._add_print()

        if self._general_par['mod_fcov']:
            print("\nModel COV ON")
        else:
            print("\nModel COV OFF")

    def _get_header(self):
        """Generate header of sim file..

        Returns
        -------
        None

        """
        header = {'obj_type': self._object_type,
                  'rate': self.rate_law[0],
                  'rate_pw': self.rate_law[1],
                  'model_name': self.sim_model.source.name}

        header = {**header, **self._general_par}
        if self.mw_dust is not None:
            header['mw_mod'] = self.mw_dust['model']
            header['mw_rv'] = self.mw_dust['rv']

        if self.vpec_dist is not None:
            header['m_vp'] = self.vpec_dist['mean_vpec']
            header['s_vp'] = self.vpec_dist['sig_vpec']

        self._update_header(header)
        return header

    @property
    def snc_model_time(self):
        """Get the sncosmo model mintime and maxtime."""
        return self.sim_model.mintime(), self.sim_model.maxtime()

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

    @time_range.setter
    def time_range(self, time_range):
        """Set the time range."""
        if time_range[0] > time_range[1]:
            print('Time range should be [Tmin,Tmax]')
        else:
            self._time_range = time_range

    @property
    def z_cdf(self):
        """Get the redshift cumulative distribution."""
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
    _available_rates = ['ptf19', 'ztf20']

    def _init_registered_rate(self):
        """SNIa rates registry."""
        if self._params['rate'].lower() == 'ptf19':
            # Rate from https://arxiv.org/abs/1903.08580
            rate = 2.43e-5 * (0.70 / self.cosmology.h)**3
            return (rate, 0)
        elif self._params['rate'].lower() == 'ztf':
            # Rate from https://arxiv.org/abs/2009.01242
            rate = 2.35e-5 * (0.70 / self.cosmology.h)**3
            return (rate, 0)

    def _init_M0(self):
        """Initialise absolute magnitude."""
        if isinstance(self._params['M0'], (float, int)):
            return self._params['M0']

        elif self._params['M0'].lower() == 'jla':
            return ut.scale_M0_jla(self.cosmology.h)

    def _init_sim_model(self):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.

        """
        if self._params['model_config']['model_name'].lower() not in self._available_models:
            raise ValueError('Model not available')

        model_dir = None
        if 'model_dir' in self._params['model_config']:
            model_dir = self._params['model_config']['model_dir']

        model = ut.init_sn_model(self._params['model_config']['model_name'],
                                 model_dir)

        if 'sct_model' in self._params:
            sct.init_sn_sct_model(model, self._params['sct_model'])
        return model

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
        # -- Generate coherent mag scattering
        astrobj_par['mag_sct'] = self.gen_coh_scatter(n_obj, seed=seed)

    def _add_print(self):
        if 'sct_model' in self._params:
            print("\nUse intrinsic scattering model : "
                  f"{self._params['sct_model']}")

    def _update_header(self, header):
        model_name = self._params['model_config']['model_name']
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

            header['mean_c'] = self._params['model_config']['dist_c'][0]

            if len(self._params['model_config']['dist_c']) == 3:
                header['dist_c'] = 'asym_gauss'
                header['sig_c_low'] = self._params['model_config']['dist_c'][1]
                header['sig_c_hi'] = self._params['model_config']['dist_c'][2]
            else:
                header['dist_c'] = 'gauss'
                header['sig_c'] = self._params['model_config']['dist_c'][1]

        if 'sct_model' in self._params:
            header['sct_mod'] = self._params['sct_model']
            if self._params['sct_model'].lower() == 'g10':
                params = ['G10_L0', 'G10_F0', 'G10_F1', 'G10_dL']
                for par in params:
                    pos = np.where(np.array(self.sim_model.param_names) == par)[0]
                    header[par] = self.sim_model.parameters[pos][0]
            elif self._params['sct_model'].lower() == 'c11':
                params = ['C11_Cuu', 'C11_Sc']
                for par in params:
                    pos = np.where(np.array(self.sim_model.param_names) == par)[0]
                    header[par] = self.sim_model.parameters[pos][0]

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
                                              z=z_for_dist)
            snc_par = [{'x1': x1, 'c': c} for x1, c in zip(sim_x1, sim_c)]

        # -- Non-coherent scattering effects
        if 'G10_' in self.sim_model.effect_names:
            seeds = rand_gen.integers(low=1e3, high=1e6, size=n_obj)
            for par, s in zip(snc_par, seeds):
                par['G10_RndS'] = s

        elif 'C11_' in self.sim_model.effect_names:
            seeds = rand_gen.integers(low=1e3, high=1e6, size=n_obj)
            for par, s in zip(snc_par, seeds):
                par['C11_RndS'] = s

        return snc_par

    def gen_salt_par(self, n_sn, seed=None, z=None):
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

        sim_c = ut.asym_gauss(*self._params['model_config']['dist_c'],
                              seed=seeds[1],
                              size=n_sn)
        return sim_x1, sim_c
