"""This module contain generators class."""

import abc
import copy
from inspect import getsource
import numpy as np
import pandas as pd
import geopandas as gpd
import sncosmo as snc
from .constants import C_LIGHT_KMS, VCMB, L_CMB, B_CMB
from . import utils as ut
from . import dust_utils as dst_ut
from . import scatter as sct
from . import salt_utils as salt_ut
from . import astrobj as astr
from . import constants as cst
from . import plasticc_model as plm


__GEN_DIC__ = {
    "snia_gen": "SNIaGen",
    "cc_gen": "CCGen",
    "snii_gen": "SNIIGen",
    "sniipl_gen": "SNIIplGen",
    "sniib_gen": "SNIIbGen",
    "sniin_gen": "SNIInGen",
    "snib/c_gen": "SNIbcGen",
    "snic_gen": "SNIcGen",
    "snib_gen": "SNIbGen",
    "snic-bl_gen": "SNIc_BLGen",
    "snia_peculiar_gen": "SNIa_peculiarGen",
    "sniax_gen": "SNIaxGen",
    "snia_91bg_gen": "SNIa_91bgGen",
    "lensed_time_series": "LensedTimeSeriesGen"
}


class BaseGen(abc.ABC):
    """Abstract class for basic astrobj generator."""

    # General attributes
    _object_type = ""
    _available_models = []  # Flux models

    def __init__(
        self,
        params,
        cosmology,
        time_range,
        z_range=None,
        vpec_dist=None,
        hosts=None,
        mw_dust=None,
        cmb=None,
        geometry=None,
    ):
        """
        Init BaseGen class.

        Parameters
        ----------
        params : dict
            Basic generator configuration.
        cosmology : astropy.cosmology
            The cosmological model to use.
        time_range : tuple
            (tmin, tmax) time range.
        z_range : tuple, optional
            (zmin, zmax) redshift range,
            no need to be defined if there is hosts, by default None
        vpec_dist : dic, optional
            PV distrib parameters, by default None

            | vpec_dist
            | ├── mean_vpec, by default 0.
            | └── sig_vpec, by default 0.
        hosts : snsim.SnHost, optional
            hosts for simulated SN, by default None
        mw_dust : dic, optional
            Milky Way dust, by default None
        cmb : dic, optional
            CMB dipole parameters, by default None

            | cmb
            | ├── v_cmb, by default 369.82 km/s
            | ├── l_cmb, by default 264.021 deg
            | └── b_cmb, by default 48.253 deg
        geometry : shapely.geometry, optional
            The survey footprint, by default None

        Raises
        ------
        ValueError
            If you set PV dist and hosts at the same time.
        ValueError
            If you neither set PV or hosts.
        ValueError
            If no hosts and no z_range.
        """

        # -- Mandatory parameters
        self._params = copy.copy(params)
        self._cosmology = cosmology
        self._time_range = time_range

        # -- At least one mandatory
        if vpec_dist is not None and hosts is not None:
            raise ValueError("You can't set vpec_dist and hosts at the same time")
        elif vpec_dist is not None and hosts is None:
            self._vpec_dist = vpec_dist
            self._hosts = None
        elif hosts is not None and vpec_dist is None:
            self._hosts = hosts
            self._vpec_dist = None
        else:
            raise ValueError("Set vpec_dist xor hosts")

        # -- If no hosts need to define a z_range
        if hosts is None:
            self._z_range = z_range
        elif hosts is not None:
            self._z_range = self.hosts._z_range
        else:
            raise ValueError("Set zrange xor hosts")

        if cmb is None:
            self._cmb = {"v_cmb": VCMB, "l_cmb": L_CMB, "b_cmb": B_CMB}
        else:
            self._cmb = cmb

        self._mw_dust = mw_dust
        self._geometry = geometry
        self.rate, self._rate_expr = self._init_rate()
        
        # -- Init absolute magnitude
        self._params['Mabs'], self._params['Mabs_band'], self._params['Mabs_str'] = self._init_absolute_mag()
        if "sigM" in self._params:
            self._params['sigM'], self._params["sigM_str"] = self._init_sigM()
            
        # -- Init sncosmo model & effects
        self.sim_sources, self._sources_prange = self._init_snc_sources()
        self.sim_effects = self._init_snc_effects()

        # -- Init redshift distribution
        self._z_dist, self._z_time_rate = self._compute_zcdf()

        # -- Get the astrobj class
        self._astrobj_class = getattr(astr, self._object_type)

    def __call__(self, n_obj=None, seed=None, basic_par=None):
        """Launch the simulation of obj.

        Parameters
        ----------
        n_obj : int
            Number of obj to simulate.
        seed : int or np.random.SeedSequence
            The random seed of the simulation.
        basic_par : pd.DataFrame
            A DataFrame that contains pre-generated parameters

        Returns
        -------
        list(AstrObj)
            A list containing Astro Object.
        """

        # -- Initialise 3 seeds for differents generation calls
        seeds = ut.gen_rndchilds(seed, 4)

        if basic_par is not None:
            n_obj = len(basic_par["zcos"])
        elif n_obj is not None:
            basic_par = self.gen_basic_par(n_obj, seed=seeds[0])
        else:
            raise ValueError("n_obj and astrobj_par cannot be None at the same time")
        
        # -- Add absolute magnitude parameters
        mag_par = {}
        mag_par["Mabs"] = np.empty(n_obj, dtype=float)
        mag_par["Mabs"].fill(self._params["Mabs"])
        mag_par["Mabs_band"] = np.empty(n_obj, dtype='U20')
        mag_par["Mabs_band"].fill(self._params["Mabs_band"])
        
        if "sigM" in self._params:
            mag_par["coh_sct"] = self.gen_coh_scatter(n_obj, seed=seeds[1])
        else:
            mag_par["coh_sct"] = np.zeros(n_obj)

        # -- Add parameters specific to the generated obj
        obj_par = self.gen_par(n_obj, basic_par, seed=seeds[2])

        # -- randomly chose the number of object for each model
        random_models = self.random_models(n_obj, seed=seeds[3])

        # -- Check if there is dust
        dust_par = {}
        if self.mw_dust is not None:
            dust_par = self._compute_dust_par(basic_par["ra"], basic_par["dec"])

        par = pd.DataFrame(
            {**random_models, **mag_par, **obj_par, **dust_par}, index=basic_par.index
            )

        par = pd.concat([basic_par, par], axis=1)

        if self.hosts is not None:
            hosts = self.hosts.df.loc[basic_par['host_index']]
            
            # -- Check for host' dust
            if "ebv" in hosts.columns:
                par["hostdust_ebv"] = hosts["ebv"].values
                if "r_v" in hosts.columns:
                    par["hostdust_r_v"] = hosts["r_v"].values
            
            # -- Check for column to keep
            if 'keep_cols' in self.hosts.config:
                for k in self.hosts.config['keep_cols']:
                    par['host_' + k] = hosts[k].values

            # --- Check for host noise columns
            if self.hosts.config['host_noise']:
                par['host_noise'] = True
                for k in hosts.columns:
                    if k.startswith('mag_') or  k.startswith('sersic_'):
                        par['host_' + k] = hosts[k].values
        
        mag_fun = None
        if "mag_fun" in self._params :
            mag_fun = self._params["mag_fun"]

        # TODO - BC: Dask that part or vectorize it for more efficiency
        return [
            self._astrobj_class(par_dic, effects=self.sim_effects, mag_fun=mag_fun)
            for par_dic in par.reset_index().to_dict(orient="records")
        ]

    def __str__(self):
        """Print config."""
        pstr = ""

        if "model_dir" in self._params:
            model_dir = self._params["model_dir"]
            model_dir_str = f" from {model_dir}"
        else:
            model_dir = None
            model_dir_str = " from sncosmo"

        pstr += "OBJECT TYPE : " + self._object_type + "\n\n"
        pstr += "SIM MODEL(S) :\n"
        for sn, snv in zip(
            self.sim_sources["model_name"], self.sim_sources["model_version"]
        ):
            pstr += f"- {sn}"
            pstr += f" v{snv}"
            pstr += model_dir_str + "\n"
        pstr += "\n"

        pstr += (
            "Peak mintime : "
            f"{self.time_range[0]:.2f} MJD\n\n"
            "Peak maxtime : "
            f"{self.time_range[1]:.2f} MJD\n\n"
        )

        pstr += "Redshift distribution computed"

        if self.hosts is not None:
            if self.hosts.config["distrib"] == "random":
                pstr += " using host redshift distribution\n"
            elif self.hosts.config["distrib"] == "survey_rate":
                pstr += " using rate\n\n"
        else:
            pstr += " using rate\n"
    
        pstr += self._add_print() + "\n"
        return pstr

    ##################################################
    # FUNCTIONS TO ADAPT FOR EACH GENERATOR SUBCLASS #
    ##################################################

    @abc.abstractmethod
    def gen_par(self, n_obj, basic_par, seed=None):
        """Abstract method to add random generated parameters
        specific to the astro object used, called in __call__

        Parameters
        ----------
        basic_par : dict(key: np.ndarray())
            Contains basic random generated properties.
        seed : int or numpy.random.SeedSequence, optional
            Random seed.
        """
        pass
    
    @abc.abstractmethod
    def _update_header(self):
        """Method to add information in header,
        called in _get_header

        Returns
        ----------
        dict
            dict is added to header dict in _get_header().
        """
        pass

    def _add_effects(self):
        """Method that return a list of effect dict.

        Notes
        -----
        Effect dict are like
        {
            'name': name of the effect,
            'source': snc.PropagationEffect subclass
            'frame': 'obs' or 'rest'
        }
        """
        return []

    def _add_print(self):
        """Method to print something in __str__."""
        pass

    def _init_sources_list(self):
        return [self._params["model_name"]]

    ####################
    # COMMON FUNCTIONS #
    ####################

    # -- INIT FUNCTIONS -- #
    def _init_absolute_mag(self):
        """Init absolute magnitude.

        Returns
        -------
        float, str, str
            Mabs values, Mabs band, Mabs input string
        """        
        if isinstance(self._params["Mabs"], (int, float)):
            Mabs = self._params["Mabs"]
            Mabs_str = None
            if "Mabs_band" in self._params:
                Mabs_band = self._params["Mabs_band"]
            else:
                if hasattr(self, '_default_Mabs_band'):
                    Mabs_band = self._default_Mabs_band
                else:
                    raise ValueError("Please provide Mabs_band parameter")
            
        elif isinstance(self._params["Mabs"], str):
            if self._object_type not in cst.Mabs_registry:
                raise ValueError(f"No default Mabs for {self._object_type} implemented yet, please provide Mabs")
            elif self._params["Mabs"].lower() not in cst.Mabs_registry[self._object_type]:
                raise ValueError(
                f"{self._params['Mabs']} is not available! Available Mabs are: {cst.Mabs_registry.keys()}"
                )
            Mabs, Mabs_band = cst.Mabs_registry[self._object_type][self._params["Mabs"].lower()]
            Mabs = ut.scale_Mabs_cosmology(
                self.cosmology.h,
                Mabs,
                cst.h_registry[self._params["Mabs"].lower().split('@')[0]],
                )
            Mabs_str = self._params["Mabs"].lower()
        else: 
            raise ValueError(f"{self._params['Mabs']} should be a float, int or str!")
        return Mabs, Mabs_band, Mabs_str
    
    def _init_sigM(self):
        if isinstance(self._params["sigM"], str):
            if self._object_type in cst.sigMabs_registery:
                if self._params["sigM"] in cst.sigMabs_registery[self._object_type]:
                    sigM = cst.sigMabs_registery[self._object_type][self._params["sigM"]]
                    sigM_str = self._params["sigM"]
                else:
                    raise ValueError(f"{self._params['sigM']} not in registery for {self._object_type}")
            else:
                raise ValueError(f"No sigM registered for {self._object_type}")
        else:
            sigM = self._params["sigM"]
            sigM_str = None
        return sigM, sigM_str
    
    def _init_registered_rate(self):
        """Rates registry.

        Returns
        -------
        str
            The rate function as a lamdba function in a str.

        Raises
        ------
        ValueError
            The rate in params is not available.
        """
        if self._params["rate"].lower() in cst.rates_registry[self._object_type]:
            expr = cst.rates_registry[self._object_type][self._params["rate"].lower()]
            registery_keys = self._params["rate"].lower().split('@')[0]
            expr += f" * ({self.cosmology.h} / {cst.h_registry[registery_keys]})**3"
        else:
            raise ValueError(
                f"{self._params['rate']} is not available! Available rate are {cst.rates_registry[self._object_type]}"
            )
        return expr
    
    def _init_snc_effects(self):
        """Init sncosmo effects.

        Returns
        -------
        list
            The list of effects dic.
        """
        effects = []
        # -- MW dust
        if self.mw_dust is not None:
            effects.append(dst_ut.init_mw_dust(self.mw_dust))
        if "ebv" in self.hosts.df.columns:
            effects.append(
                {
                    'source': snc.CCM89Dust(),
                    'name': 'hostdust_',
                    'frame': 'rest'
                }
            )
        effects += self._add_effects()
        return effects

    def _init_snc_sources(self):
        """
        Init the sncosmo model.

        Returns
        -------
        dic
            Sources dic.

        Raises
        ------
        ValueError
            model_name not available.
        """
        # -- Check existence of the model
        if isinstance(self._params["model_name"], str) & (
            self._params["model_name"] not in self._available_models
        ):
            raise ValueError(f"{self._params['model_name']} is not available")
        elif isinstance(self._params["model_name"], list):
            for s in self._params["model_name"]:
                if s not in self._available_models:
                    raise ValueError(f"{s} is not available")

        sources = {"model_name": self._init_sources_list()}

        if "model_version" in self._params:
            if not isinstance(self._params["model_version"], list):
                sources["model_version"] = [self._params["model_version"]]
        else:
            sources["model_version"] = [None] * len(sources["model_name"])

        # -- Compute max, min phase
        snc_sources = [
            snc.get_source(name=n, version=v)
            for n, v in zip(sources["model_name"], sources["model_version"])
        ]

        sources["model_version"] = [s.version for s in snc_sources]
        maxphase = np.max([s.maxphase() for s in snc_sources])
        minphase = np.min([s.minphase() for s in snc_sources])
        return sources, (minphase, maxphase)

    def _init_rate(self):
        """Initialise rate in obj/Mpc^-3/year
        Returns
        -------
            lambda funtion, str
                The funtion and it's expression as a string
        """
        if "rate" in self._params:
            if isinstance(self._params["rate"], type(lambda: 0)):
                rate = self._params["rate"]
                expr = "".join(
                    getsource(self._params["rate"]).partition("lambda")[1:]
                ).replace(",", "")
            elif isinstance(self._params["rate"], str):
                # Check for lambda function in str
                if "lambda" in self._params["rate"].lower():
                    expr = self._params["rate"]
                # Check registered rate
                elif self._params["rate"].lower() in cst.rates_registry[self._object_type]:
                    expr = self._init_registered_rate()
                # Check for yaml bad conversion of '1e-5'
                else:
                    expr = f"lambda z: {float(self._params['rate'])}"
            else:
                expr = f"lambda z: {self._params['rate']}"
        # Default
        else:
            expr = "lambda z: 3e-5"
        return eval(expr), expr.strip()

    def _compute_zcdf(self):
        """Give the time rate SN/years in redshift shell.

        Returns
        -------
        snsim.utils.CustomRandom, (float, float)
            Redshift dist, (shell redshifts, shell rates)

        """
        z_min, z_max = self.z_range

        # -- Set the precision to dz = 1e-5
        dz = 1e-5

        z_shell = np.linspace(z_min, z_max, int((z_max - z_min) / dz))
        dz = z_shell[1] - z_shell[0]

        co_dist = self.cosmology.comoving_distance(z_shell).value
        shell_vol = (
            4 * np.pi * co_dist**2 * C_LIGHT_KMS / self.cosmology.H(z_shell).value * dz
        )

        # -- Compute the sn time rate in each volume shell [( SN / year )(z)]
        shell_time_rate = self.rate(z_shell) * shell_vol / (1 + z_shell)

        z_pdf = lambda x: np.interp(x, z_shell, shell_time_rate)

        return ut.CustomRandom(z_pdf, z_min, z_max, dx=1e-5), (z_shell, shell_time_rate)

    def _compute_dust_par(self, ra, dec):
        """Compute dust parameters.
        Parameters
        ----------
        ra : numpy.ndaray(float)
            SN Right Ascension rad.
        dec : numpy.ndarray(float)
            SN Declinaison rad.
        Returns
        -------
        list(dict)
            List of Dictionnaries that contains Rv and E(B-V) for each SN.
        """
        mod_name = self.mw_dust["model"]
        dust_par = {"mw_ebv": dst_ut.compute_ebv(ra, dec)}

        if mod_name.lower() in ["ccm89", "od94"]:
            if "rv" not in self.mw_dust:
                self.mw_dust["rv"] = 3.1
            dust_par["mw_r_v"] = np.ones(len(ra)) * self.mw_dust["rv"]
        return dust_par

    def _get_header(self):
        """Generate header of sim file."""
        header = {
            "obj_type": self._object_type,
            "rate": self._rate_expr,
            "params": self._params,
            **self.sim_sources,
        }

        if self.vpec_dist is not None:
            header["vpec_dist"] = self.vpec_dist

        if self.mw_dust is not None:
            header['mw_dust'] = self.mw_dust

        header = {**header, **self._update_header()}
        return header

    # -- RANDOM FUNCTIONS -- #

    def gen_peak_time(self, n, seed=None):
        """Generate uniformly n peak time in the survey time range.

        Parameters
        ----------
        n : int
            Number of time to generate.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None

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
            Number of coords to generate.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None

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

                multipoint = gpd.points_from_xy(ra_tmp, dec_tmp)
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
            Number of redshifts to generate.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None

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
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing vpec (km/s) generated.

        """
        rand_gen = np.random.default_rng(seed)

        vpec = rand_gen.normal(
            loc=self.vpec_dist["mean_vpec"], scale=self.vpec_dist["sig_vpec"], size=n
        )
        return vpec

    def gen_basic_par(self, n_obj, seed=None, min_max_t=False):
        """Generate basic obj properties.

        Parameters
        ----------
        n_obj: int
            Number of obj.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None

        Notes
        -----
        List of parameters:
            * t0 : obj peak
            * zcos : cosmological redshift
            * ra : Right Ascension
            * dec : Declinaison
            * vpec : peculiar velocity
            * como_dist : comoving distance
            * zpcmb : CMB dipole redshift contribution
            * mw_ebv, opt : Milky way dust extinction
            * host_, opt : host parameters
        """
        # -- Generate seeds for random calls
        seeds = ut.gen_rndchilds(seed, 4)

        # -- Generate peak time
        t0 = self.gen_peak_time(n_obj, seed=seeds[0])

        if self.hosts is None:
            # -- Generate cosmological redshifts
            zcos = self.gen_zcos(n_obj, seed=seeds[1])
            
            # -- Generate ra, dec
            ra, dec = self.gen_coord(n_obj, seed=seeds[2])
            
            # -- Generate vpec
            if self.vpec_dist is not None:
                vpec = self.gen_vpec(n_obj, seed=seeds[3])
            else:
                vpec = np.zeros(len(ra))
        else:
            # -- Draw hosts
            hosts = self.hosts.random_choice(
                n_obj,
                seed=seeds[1],
                rate=self.rate,
                sn_type=self._object_type,
                cosmology=self.cosmology,
            )
            
            zcos = hosts["zcos"].values
            ra = hosts["ra"].values
            dec = hosts["dec"].values
            vpec = hosts["vpec"].values

        basic_par = {
            "zcos": zcos,
            "como_dist": self.cosmology.comoving_distance(zcos).value,
            "zpcmb": ut.compute_zpcmb(ra, dec, self.cmb),
            "t0": t0,
            "ra": ra,
            "dec": dec,
            "vpec": vpec,
        }

        if min_max_t:
            _1_zobs_ = 1 + basic_par["zcos"]
            _1_zobs_ *= 1 + basic_par["zpcmb"]
            _1_zobs_ *= 1 + basic_par["vpec"] / C_LIGHT_KMS
            basic_par["min_t"] = basic_par["t0"] + self._sources_prange[0] * _1_zobs_
            basic_par["max_t"] = basic_par["t0"] + self._sources_prange[1] * _1_zobs_
            basic_par["1_zobs"] = _1_zobs_

        if self.hosts is not None:
            basic_par['host_index'] = hosts.index

        return pd.DataFrame(basic_par)
    
    def gen_coh_scatter(self, n_obj, seed=None):
        rand_gen = np.random.default_rng(seed)
        
        if isinstance(self._params["sigM"], (float, int)):
            return rand_gen.normal(loc=0, scale=self._params["sigM"], size=n_obj)

        elif isinstance(self._params["sigM"], list):
            return ut.asym_gauss(
                mu=0,
                sig_low=self._params["sigM"][0],
                sig_high=self._params["sigM"][1],
                seed=seed,
                size=n_obj,
            )
        
    def random_models(self, n_obj, seed=None):
        """Draw n random models for a given source.

        Parameters
        ----------
        n_obj : int
            Number of models to draw.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None

        Returns
        -------
        dic(model_names: list, model_version: list)
            Dic which contains list of model_names and versions.
        """
        rand_gen = np.random.default_rng(seed)

        idx = rand_gen.integers(
            low=0, high=len(self.sim_sources["model_name"]), size=n_obj
        )
        random_models = {
            "model_name": np.array(self.sim_sources["model_name"])[idx],
            "model_version": np.array(self.sim_sources["model_version"])[idx],
        }
        return random_models

    @property
    def hosts(self):
        """Get the host class."""
        return self._hosts

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
    """SNIa parameters generator. Inherit from BaseGen"""

    _object_type = "SNIa"
    _default_Mabs_band = ['bessellb']
    _available_models = ["salt2", "salt3"]
    
    def _add_print(self):
        """Add print statement."""
        str = ""
        if "sct_model" in self._params:
            str += "\nUse intrinsic scattering model : " f"{self._params['sct_model']}"
        return str

    def _add_effects(self):
        effects = []
        args = []
        # Add scattering model if needed
        if "sct_model" in self._params:
            if self._params["sct_model"] == "G10":
                if len(self.sim_sources["model_name"]) > 1 or self.sim_sources[
                    "model_name"
                ][0] not in ["salt2", "salt3"]:
                    raise ValueError("G10 cannot be used")
                args = [
                    snc.get_source(
                                name=self.sim_sources["model_name"][0],
                                version=self.sim_sources["model_version"][0],
                            )
                        ]
            effects.append(sct.init_sn_sct_model(self._params["sct_model"], *args))
        return effects

    def _update_header(self):
        model_name = self._params["model_name"]

        header = {}
        if model_name.lower()[:4] == "salt":
            if isinstance(self._params["dist_x1"], str):
                header["dist_x1"] = self._params["dist_x1"]
            else:
                header["peak_x1"] = self._params["dist_x1"][0]
                if len(self._params["dist_x1"]) == 3:
                    header["dist_x1"] = "asym_gauss"
                    header["sig_x1_low"] = self._params["dist_x1"][1]
                    header["sig_x1_hi"] = self._params["dist_x1"][2]
                elif len(self._params["dist_x1"]) == 2:
                    header["dist_x1"] = "gauss"
                    header["sig_x1"] = self._params["dist_x1"][1]

            if isinstance(self._params["dist_c"], str):
                if self._params["dist_c"].lower() == "bs20":
                    header["peak_c"] = "BS20"
                    header["dist_c"] = "c_int BS20"
                    header["sig_c"] = "c_int BS20"

            elif isinstance(self._params["dist_c"], list):
                if len(self._params["dist_c"]) == 3:
                    header["peak_c"] = self._params["dist_c"][0]
                    header["dist_c"] = "asym_gauss"
                    header["sig_c_low"] = self._params["dist_c"][1]
                    header["sig_c_hi"] = self._params["dist_c"][2]
                else:
                    header["peak_c"] = self._params["dist_c"][0]
                    header["dist_c"] = "gauss"
                    header["sig_c"] = self._params["dist_c"][1]
        return header

    def gen_par(self, n_obj, basic_par, seed=None):
        """Generate SNIa specific parameters.

        Parameters
        ----------
        n_obj : int
            Number of parameters to generate.
        basic_par: pd.DataFrame
            Dataframe with pre-generated parameters.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None

        Returns
        -------
        dict
            One dictionnary containing 'parameters names': numpy.ndarray(float).

        """
        seeds = ut.gen_rndchilds(seed=seed, size=3)
        params = {}
        
        # -- Spectra model parameters
        if self._params["model_name"] in ("salt2", "salt3"):
            sim_x1, sim_c, alpha, beta = self.gen_salt_par(
                n_obj, seeds[1], basic_par=basic_par
            )
            params = {**params, "x1": sim_x1, "c": sim_c, "alpha": alpha, "beta": beta}

        # -- Mass step
        mass_step = np.zeros(n_obj)
        if "mass_step" in self._params:
            mask = np.log10(self.hosts.df.loc[basic_par['host_index']]['sm']) >  self._params["mass_step"][0]
            mass_step[mask] = self._params["mass_step"][1] / 2 
            mass_step[~mask] = -self._params["mass_step"][1] / 2 
        
        params["mass_step"] = mass_step

        # -- Non-coherent scattering effects
        if "sct_model" in self._params:
            randgen = np.random.default_rng(seeds[2])
            if self._params["sct_model"] == "G10":
                params["G10_RndS"] = randgen.integers(1e12, size=n_obj)
            elif self._params["sct_model"] == "C11":
                params["C11_RndS"] = randgen.integers(1e12, size=n_obj)
            elif self._params["sct_model"].lower() == "bs20":
                params["BS20_r_v"], params["BS20_ebv"] = sct.gen_BS20_scatter(
                    n_obj, par_names=['Rv', 'E_dust'], seed=seeds[2]
                )

        return params

    def gen_salt_par(self, n_sn, seed=None, basic_par=None):
        """Generate SALT parameters.

        Parameters
        ----------
        n_sn : int
            Number of parameters to generate.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None
        basic_par : pd.DataFrame
            Pre-generated parameters.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing SALT2 x1 and c generated parameters.

        """
        seeds = ut.gen_rndchilds(seed=seed, size=4)

        # -- x1 dist
        if isinstance(self._params["dist_x1"], str):
            if self._params["dist_x1"].lower() == "n21":
                sim_x1 = salt_ut.n21_x1_model(basic_par["zcos"], seed=seeds[0])
            elif self._params["dist_x1"].lower() == "n21+mass":
                sim_x1 = salt_ut.n21_x1_mass_model(
                    basic_par["zcos"], self.hosts.loc[basic_par["host_index"]]['sm'].values, seed=seeds[0]
                )
            elif self._params["dist_x1"].lower() == "mass":
                sim_x1 = salt_ut.x1_mass_model(self.hosts.df.loc[basic_par["host_index"]]['sm'].values, seed=seeds[0])

        elif isinstance(self._params["dist_x1"], list):
            sim_x1 = ut.asym_gauss(*self._params["dist_x1"], seed=seeds[0], size=n_sn)

        # -- c dist
        if isinstance(self._params["dist_c"], str):
            if self._params["dist_c"].lower() == "bs20":
                sim_c = sct.gen_BS20_scatter(n_sn, par_names='c_int', seed=seeds[1])[0]
        else:
            sim_c = ut.asym_gauss(*self._params["dist_c"], seed=seeds[1], size=n_sn)

        # -- alpha dist
        if isinstance(self._params["alpha"], float):
            alpha = np.ones(n_sn) * self._params["alpha"]
        elif isinstance(self._params["alpha"], list):
            alpha = ut.asym_gauss(*self._params["alpha"], seed=seeds[2], size=n_sn)
        # -- beta dist
        if isinstance(self._params["beta"], float):
            beta = np.ones(n_sn) * self._params["beta"]
        elif isinstance(self._params["alpha"], list):
            beta = ut.asym_gauss(*self._params["beta"], seed=seeds[3], size=n_sn)
        elif isinstance(self._params["beta"], str):
            if self._params["beta"].lower() == "bs20":
                beta = sct.gen_BS20_scatter(n_sn, par_names='beta_sn', seed=seeds[3])[0]
        return sim_x1, sim_c, alpha, beta


class CCGen(BaseGen):
    """Template for CoreColapse. Inherit from BaseGen.

    Notes
    -----

    For Rate:
        * SNCC ztf20 relative fraction of SNe subtypes from https://arxiv.org/abs/2009.01242 figure 6 +
        ztf20 relative fraction between SNe Ic and SNe Ib from https://iopscience.iop.org/article/10.3847/1538-4357/aa5eb7/meta
        * SNCC shiver17 fraction from https://arxiv.org/abs/1609.02922 Table 3

    For Luminosity Functions:
        * SNCC Mabs mean and scattering of luminosity function values from Vincenzi et al. 2021 Table 5 (https://arxiv.org/abs/2111.10382)
    """

    _available_models = ["vin19_corr", "vin19_nocorr"]

    def gen_par(self, n_obj, basic_par, seed=None):
        """Generate sncosmo model dependant parameters (others than redshift and t0).
        Parameters
        ----------
        n_obj : int
            Number of parameters to generate.
        basic_par :
            Pre-generated parameters.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None
            .
        Returns
        -------
        dict
            One dictionnary containing 'parameters names': numpy.ndarray(float).
        """
        return {}

    def _add_print(self):
        str = ""
        return str

    def _update_header(self):
        return {}

    def _init_sources_list(self):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.
        """
        if isinstance(self._params["model_name"], str):
            if self._params["model_name"].lower() == "all":
                sources = self._available_models
            elif self._params["model_name"].lower() == "vin19_nocorr":
                sources = ut.select_Vincenzi_template(
                    self._available_models, corr=False
                )
            elif self._params["model_name"].lower() == "vin19_corr":
                sources = ut.select_Vincenzi_template(self._available_models, corr=True)
            else:
                sources = [self._params["model_name"]]
        else:
            sources = self._params["model_name"]
        return sources


class SNIIGen(CCGen):
    """SNII parameters generator. Inherit from CCGen."""

    _object_type = "SNII"
    _available_models = ut.Templatelist_fromsncosmo("snii") + CCGen._available_models


class SNIIplGen(CCGen):
    """SNIIPL parameters generator. Inherit from CCGen."""

    _object_type = "SNIIpl"
    _available_models = ut.Templatelist_fromsncosmo("sniipl") + CCGen._available_models


class SNIIbGen(CCGen):
    """SNIIb parameters generator. Inherit from CCGen."""

    _object_type = "SNIIb"
    _available_models = ut.Templatelist_fromsncosmo("sniib") + CCGen._available_models


class SNIInGen(CCGen):
    """SNIIn parameters generator. Inherit from CCGen."""

    _object_type = "SNIIn"
    _available_models = ut.Templatelist_fromsncosmo("sniin") + CCGen._available_models


class SNIbcGen(CCGen):
    """SNIb/c parameters generator. Inherit from CCGen."""

    _object_type = "SNIb/c"
    _available_models = ut.Templatelist_fromsncosmo("snib/c") + CCGen._available_models


class SNIcGen(CCGen):
    """SNIc class. Inherit from CCGen."""

    _object_type = "SNIc"
    _available_models = ut.Templatelist_fromsncosmo("snic") + CCGen._available_models


class SNIbGen(CCGen):
    """SNIb class. Inherit from CCGen."""

    _object_type = "SNIb"
    _available_models = ut.Templatelist_fromsncosmo("snib") + CCGen._available_models
    

class SNIc_BLGen(CCGen):
    """SNIc_BL class. Inherit from CCGen."""

    _object_type = "SNIc_BL"
    _available_models = ut.Templatelist_fromsncosmo("snic-bl") + CCGen._available_models


class SNIapeculiarGen(BaseGen):
    """SNIa_peculiar class.

     Models form platicc challenge ask Rick
     need a directory to store model

     Parameters
     ----------
    same as TimeSeriesGen class"""

    _available_models = ["plasticc"]

    def _init_sources_list(self):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.
        """
        if isinstance(self._params["model_name"], str):
            if self._params["model_name"].lower() == "plasticc":
                sources = plm.get_sed_listname(self._object_type.lower())
            else:
                sources = [self._params["model_name"]]
        else:
            sources = self._params["model_name"]

        return sources

    def gen_par(self, n_obj, basic_par, seed=None):

        params = {}

        if self._object_type.lower() == "sniax":
            rv, e_dust = self._gen_dust_par(n_obj, seed)

            params["E_dust"] = e_dust
            params["RV"] = rv

        return params

    def _add_print(self):
        return ""

    def _update_header(self):
        return {}


class SNIaxGen(SNIapeculiarGen):
    """SNIaxclass.

     Models form platicc challenge ask Rick
     need a directory to store model

     Parameters
     ----------
    same as TimeSeriesGen class"""

    _object_type = "SNIax"
    _available_models = (
        plm.get_sed_listname("snia91bg") + SNIapeculiarGen._available_models
    )
    
    def _gen_dust_par(self, n_obj, seed):
        return plm.generate_dust_sniax(n_obj, seed)


class SNIa_91bgGen(SNIapeculiarGen):
    """SNIa 91bg-like class.

     Models form platicc challenge ask Rick
     need a directory to store model

     Parameters
     ----------
    same as TimeSeriesGen class"""

    _object_type = "SNIa91bg"
    _available_models = (
        plm.get_sed_listname("snia91bg") + SNIapeculiarGen._available_models
    )
