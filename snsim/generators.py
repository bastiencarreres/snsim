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
}


class BaseGen(abc.ABC):
    """Abstract class for basic astrobj generator."""

    # General attributes
    _object_type = ""
    _available_models = []  # Flux models
    _available_rates = {}  # Rate models

    def __init__(
        self,
        params,
        cosmology,
        time_range,
        z_range=None,
        vpec_dist=None,
        host=None,
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
            no need to be defined if there is host, by default None
        vpec_dist : dic, optional
            PV distrib parameters, by default None
            | vpec_dist
            | ├── mean_vpec, by default 0.
            | └── sig_vpec, by default 0.
        host : snsim.SnHost, optional
            host for simulated SN, by default None
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
            If you set PV dist and host at the same time.
        ValueError
            If you neither set PV or host.
        ValueError
            If no host and no z_range.
        """

        # -- Mandatory parameters
        self._params = copy.copy(params)
        self._cosmology = cosmology
        self._time_range = time_range

        # -- At least one mandatory
        if vpec_dist is not None and host is not None:
            raise ValueError("You can't set vpec_dist and host at the same time")
        elif vpec_dist is not None and host is None:
            self._vpec_dist = vpec_dist
            self._host = None
        elif host is not None and vpec_dist is None:
            self._host = host
            self._vpec_dist = None
        else:
            raise ValueError("Set vpec_dist xor host")

        # -- If no host need to define a z_range
        if host is None:
            self._z_range = z_range
        elif host is not None:
            self._z_range = self.host._z_range
        else:
            raise ValueError("Set zrange xor host")

        if cmb is None:
            self._cmb = {"v_cmb": VCMB, "l_cmb": L_CMB, "b_cmb": B_CMB}
        else:
            self._cmb = cmb

        self._mw_dust = mw_dust
        self._geometry = geometry
        self.rate, self._rate_expr = self._init_rate()

        # -- Init sncosmo model & effects
        self.sim_sources, self._sources_prange = self._init_snc_sources()
        self.sim_effects = self._init_snc_effects()

        # -- Init redshift distribution
        self._z_dist, self._z_time_rate = self._compute_zcdf()

        # -- Get the astrobj class
        self._astrobj_class = getattr(astr, self._object_type)

        # if peak_out_trange:
        #     t0 = self.time_range[0] - self.snc_model_time[1] * (1 + self.z_range[1])
        #     t1 = self.time_range[1] - self.snc_model_time[0] * (1 + self.z_range[1])
        #     self._time_range = (t0, t1)

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
        seeds = ut.gen_rndchilds(seed, 3)

        if basic_par is not None:
            n_obj = len(basic_par["zcos"])
        elif n_obj is not None:
            basic_par = self.gen_basic_par(n_obj, seed=seeds[0])
        else:
            raise ValueError("n_obj and astrobj_par cannot be None at the same time")

        # -- Add parameters specific to the generated obj
        obj_par = self.gen_par(n_obj, basic_par, seed=seeds[1])

        # -- randomly chose the number of object for each model
        random_models = self.random_models(n_obj, seed=seeds[2])

        # -- Check if there is dust
        if self.mw_dust is not None:
            dust_par = self._compute_dust_par(basic_par["ra"], basic_par["dec"])
        else:
            dust_par = {}

        par = pd.DataFrame(
            {**random_models, **obj_par, **dust_par}, index=basic_par.index
        )

        par = pd.concat([basic_par, par], axis=1)

        if "relation" not in self._params:
            relation = None
        else:
            relation = self._params["relation"]

        # TODO - BC: Dask that part or vectorize it for more efficiency
        return [
            self._astrobj_class(par_dic, effects=self.sim_effects, relation=relation)
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

        pstr += "OBJECT TYPE : " + self._object_type + "\n"
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

        if self.host is not None:
            if self.host.config["distrib"] == "random":
                pstr += " using host redshift distribution\n"
            elif self.host.config["distrib"] == "survey_rate":
                pstr += " using rate\n\n"
        else:
            pstr += " using rate\n"

        pstr += self._add_print() + "\n\n"
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
        if self._params["rate"].lower() in self._available_rates:
            return self._available_rates[self._params["rate"].lower()].format(
                h=self.cosmology.h
            )
        else:
            raise ValueError(
                f"{self._params['rate']} is not available! Available rate are {self._available_rates}"
            )

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
        if (
            self._object_type.lower() == "sniax"
            or self._object_type.lower() == "snia91bg"
        ):
            snc_sources = [
                plm.snc_source_from_sed(name, self._sed_path)
                for name in sources["model_name"]
            ]

        else:
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
                elif self._params["rate"].lower() in self._available_rates:
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
            if "r_v" in self.mw_dust:
                rv_val = self.mw_dust["r_v"]
            else:
                rv_val = 3.1
            dust_par["mw_r_v"] = np.ones(len(ra)) * rv_val
        return dust_par

    def _get_header(self):
        """Generate header of sim file."""
        header = {
            "obj_type": self._object_type,
            "rate": self._rate_expr,
            **self.sim_sources,
        }

        if self.vpec_dist is not None:
            header["m_vp"] = self.vpec_dist["mean_vpec"]
            header["s_vp"] = self.vpec_dist["sig_vpec"]

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

        # -- Generate cosmological redshifts
        if self.host is None:
            zcos = self.gen_zcos(n_obj, seed=seeds[1])
        else:
            host = self.host.random_choice(
                n_obj,
                seed=seeds[1],
                rate=self.rate,
                sn_type=self._object_type,
                cosmology=self.cosmology,
            )
            zcos = host["zcos"].values

        # -- Generate ra, dec
        if self.host is not None:
            ra = host["ra"].values
            dec = host["dec"].values
        else:
            ra, dec = self.gen_coord(n_obj, seed=seeds[2])

        # -- Generate vpec
        if self.vpec_dist is not None:
            vpec = self.gen_vpec(n_obj, seed=seeds[3])
        elif self.host is not None:
            vpec = host["vpec"].values
        else:
            vpec = np.zeros(len(ra))

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

        if "mass_step" in self._params:
            basic_par["mass_step"] = self._params["mass_step"]

        # Save in basic_par all the column of the host_table that start with host, to save the data in final sim
        if self.host is not None:
            basic_par["host_index"] = host.index
            for k in host.columns:
                if k.startswith("host_"):
                    basic_par[k] = host[k].values

        return pd.DataFrame(basic_par)

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
    """SNIa parameters generator. Inherit from BaseGen"""

    _object_type = "SNIa"
    _available_models = ["salt2", "salt3"]
    _available_rates = {
        "ptf19": "lambda z:  2.43e-5 * ({h}/0.70)**3",  # Rate from https://arxiv.org/abs/1903.08580
        "ztf20": "lambda z:  2.35e-5 * ({h}/0.70)**3",  # Rate from https://arxiv.org/abs/2009.01242
        "ptf19_pw": "lambda z:  2.35e-5 * ({h}/0.70)**3 * (1 + z)**1.7",  # Rate from https://arxiv.org/abs/1903.08580
    }

    SNIA_M0 = {
        "jla": -19.05
    }  # M0 SNIA from JLA paper (https://arxiv.org/abs/1401.4064)

    def _init_M0(self):
        """Initialise absolute magnitude."""
        if isinstance(self._params["M0"], (float, np.floating, int, np.integer)):
            return self._params["M0"]
        elif self._params["M0"].lower() in self.SNIA_M0:
            return ut.scale_M0_cosmology(
                self.cosmology.h,
                self.SNIA_M0[self._params["M0"].lower()],
                cst.h_article[self._params["M0"].lower()],
            )
        else:
            raise ValueError(
                f"{self._params['M0']} is not available! Available M0 are {self.SNIA_M0.keys()}"
            )

    def _add_print(self):
        """Add print statement."""
        str = ""
        if "sct_model" in self._params:
            str += "\nUse intrinsic scattering model : " f"{self._params['sct_model']}"
        return str

    def _add_effects(self):
        effects = []
        # Add scattering model if needed
        if "sct_model" in self._params:
            if self._params["sct_model"] == "G10":
                if len(self.sim_sources["model_name"]) > 1 or self.sim_sources[
                    "model_name"
                ][0] not in ["salt2", "salt3"]:
                    raise ValueError("G10 cannot be used")
                effects.append(
                    {
                        "source": sct.G10(
                            snc.get_source(
                                name=self.sim_sources["model_name"][0],
                                version=self.sim_sources["model_version"][0],
                            )
                        ),
                        "frame": "rest",
                        "name": "G10_",
                    }
                )
            elif self._params["sct_model"] in ["C11_0", "C11_1", "C11_2"]:
                effects.append({"source": sct.C11(), "frame": "rest", "name": "C11_"})

        return effects

    def _update_header(self):
        model_name = self._params["model_name"]

        header = {}
        header["M0_band"] = "bessell_b"
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

        params = {
            "M0": np.ones(n_obj) * self._init_M0(),
            "coh_sct": self.gen_coh_scatter(n_obj, seed=seeds[0]),
        }

        # -- Non-coherent scattering effects
        if "sct_model" in self._params:
            randgen = np.random.default_rng(seeds[2])
            if self._params["sct_model"] == "G10":
                params["G10_RndS"] = randgen.integers(1e12, size=n_obj)
            elif self._params["sct_model"] == "C11":
                params["C11_RndS"] = randgen.integers(1e12, size=n_obj)
            elif self._params["sct_model"].lower() == "bs20":
                _, params["RV"], params["E_dust"], _ = sct.gen_BS20_scatter(
                    n_obj, seeds[2]
                )

        # -- Spectra model parameters
        model_name = self._params["model_name"]

        if model_name in ("salt2", "salt3"):
            sim_x1, sim_c, alpha, beta = self.gen_salt_par(
                n_obj, seeds[1], basic_par=basic_par
            )
            params = {**params, "x1": sim_x1, "c": sim_c, "alpha": alpha, "beta": beta}

        return params

    def gen_coh_scatter(self, n_sn, seed=None):
        """Generate n coherent mag scattering term.

        Parameters
        ----------
        n_sn : int
            Number of mag scattering terms to generate.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing scattering terms generated.

        """
        rand_gen = np.random.default_rng(seed)

        mag_sct = rand_gen.normal(loc=0, scale=self._params["sigM"], size=n_sn)
        return mag_sct

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
                    basic_par["zcos"], basic_par["host_mass"], seed=seeds[0]
                )
            elif self._params["dist_x1"].lower() == "mass":
                sim_x1 = salt_ut.x1_mass_model(basic_par["host_mass"], seed=seeds[0])

        elif isinstance(self._params["dist_x1"], list):
            sim_x1 = ut.asym_gauss(*self._params["dist_x1"], seed=seeds[0], size=n_sn)

        # -- c dist
        if isinstance(self._params["dist_c"], str):
            if self._params["dist_c"].lower() == "bs20":
                _, _, _, sim_c = sct.gen_BS20_scatter(n_sn, seeds[1])
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
                beta, _, _, _ = sct.gen_BS20_scatter(n_sn, seeds[3])
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
        * SNCC M0 mean and scattering of luminosity function values from Vincenzi et al. 2021 Table 5 (https://arxiv.org/abs/2111.10382)
    """

    _available_models = ["vin19_corr", "vin19_nocorr"]

    def _init_M0(self):
        """Initialise absolute magnitude."""
        if isinstance(self._params["M0"], (float, np.floating, int, np.integer)):
            return self._params["M0"]
        else:
            return self.init_M0_for_type()

    def gen_coh_scatter(self, n_sn, seed=None):
        """Generate n coherent mag scattering term.

        Parameters
        ----------
        n : int
            Number of mag scattering terms to generate.
        seed : int or numpy.random.SeedSequence, optional
            Random seed, by default None

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing scattering terms generated.

        """
        rand_gen = np.random.default_rng(seed)

        if isinstance(self._params["sigM"], (float, np.floating, int, np.integer)):
            return rand_gen.normal(loc=0, scale=self._params["sigM"], size=n_sn)

        elif isinstance(self._params["sigM"], list):
            return ut.asym_gauss(
                mu=0,
                sig_low=self._params["sigM"][0],
                sig_high=self._params["sigM"][1],
                seed=seed,
                size=n_sn,
            )
        else:
            return self.gen_coh_scatter_for_type(n_sn, seed)

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
        params = {
            "M0": np.ones(n_obj) * self._init_M0(),
            "coh_sct": self.gen_coh_scatter(n_obj, seed=seed),
        }
        return params

    def _add_print(self):
        str = ""
        return str

    def _update_header(self):
        header = {}
        header["M0_band"] = "bessell_r"
        return header

    def init_M0_for_type(self):
        """Initialise absolute magnitude using default values from past literature works based on the type."""
        if self._params["M0"].lower() == "li11_gaussian":
            return ut.scale_M0_cosmology(
                self.cosmology.h,
                self._sn_lumfunc["M0"]["li11_gaussian"],
                cst.h_article["li11"],
            )

        elif self._params["M0"].lower() == "li11_skewed":
            return ut.scale_M0_cosmology(
                self.cosmology.h,
                self._sn_lumfunc["M0"]["li11_skewed"],
                cst.h_article["li11"],
            )
        else:
            raise ValueError(
                f"{self._params['M0']} is not available! Available M0 are {self._sn_lumfunc['M0'].keys()} "
            )

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

    def gen_coh_scatter_for_type(self, n_sn, seed):
        """Generate n coherent mag scattering term using default values from past literature works based on the type."""
        if self._params["sigM"].lower() == "li11_gaussian":
            return ut.asym_gauss(
                mu=0,
                sig_low=self._sn_lumfunc["coh_sct"]["li11_gaussian"][0],
                sig_high=self._sn_lumfunc["coh_sct"]["li11_gaussian"][1],
                seed=seed,
                size=n_sn,
            )

        elif self._params["sigM"].lower() == "li11_skewed":
            return ut.asym_gauss(
                mu=0,
                sig_low=self._sn_lumfunc["coh_sct"]["li11_skewed"][0],
                sig_high=self._sn_lumfunc["coh_sct"]["li11_skewed"][1],
                seed=seed,
                size=n_sn,
            )
        else:
            raise ValueError(
                f"{self._params['sigM']} is not available! Available sigM are {self._sn_lumfunc['coh_scatter'].keys()} "
            )


class SNIIGen(CCGen):
    """SNII parameters generator. Inherit from CCGen."""

    _object_type = "SNII"
    _available_models = ut.Templatelist_fromsncosmo("snii") + CCGen._available_models

    _sn_fraction = {"ztf20": 0.776208, "shivers17": 0.69673}

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3  * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6",
    }

    def init_M0_for_type(self):
        raise ValueError("Default M0 for SNII not implemented yet, please provide M0")

    def gen_coh_scatter_for_type(self, n_sn, seed):
        raise ValueError(
            "Default scatterting for SNII not implemented yet, please provide SigM"
        )


class SNIIplGen(CCGen):
    """SNIIPL parameters generator. Inherit from CCGen."""

    _object_type = "SNIIpl"
    _available_models = ut.Templatelist_fromsncosmo("sniipl") + CCGen._available_models

    _sn_lumfunc = {
        "M0": {"li11_gaussian": -15.97, "li11_skewed": -17.51},
        "coh_sct": {"li11_gaussian": [1.31, 1.31], "li11_skewed": [2.01, 3.18]},
    }

    _sn_fraction = {
        "shivers17": 0.620136,
        "ztf20": 0.546554,
    }

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)",
    }


class SNIIbGen(CCGen):
    """SNIIb parameters generator. Inherit from CCGen."""

    _object_type = "SNIIb"
    _available_models = ut.Templatelist_fromsncosmo("sniib") + CCGen._available_models
    _available_rates = ["ptf19", "ztf20", "ptf19_pw"]
    _sn_lumfunc = {
        "M0": {"li11_gaussian": -16.69, "li11_skewed": -18.30},
        "coh_sct": {"li11_gaussian": [1.38, 1.38], "li11_skewed": [2.03, 7.40]},
    }

    _sn_fraction = {
        "shivers17": 0.10944,
        "ztf20": 0.047652,
    }

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)",
    }


class SNIInGen(CCGen):
    """SNIIn parameters generator. Inherit from CCGen."""

    _object_type = "SNIIn"
    _available_models = ut.Templatelist_fromsncosmo("sniin") + CCGen._available_models

    _sn_lumfunc = {
        "M0": {"li11_gaussian": -17.90, "li11_skewed": -19.13},
        "coh_sct": {"li11_gaussian": [0.95, 0.95], "li11_skewed": [1.53, 6.83]},
    }

    _sn_fraction = {
        "shivers17": 0.046632,
        "ztf20": 0.102524,
    }

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)",
    }


class SNIbcGen(CCGen):
    """SNIb/c parameters generator. Inherit from CCGen."""

    _object_type = "SNIb/c"
    _available_models = ut.Templatelist_fromsncosmo("snib/c") + CCGen._available_models
    _sn_fraction = {"ztf20": 0.217118, "shivers17": 0.19456}

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)",
    }

    def init_M0_for_type(self):
        raise ValueError("Default M0 for SNII not implemented yet, please provide M0")

    def gen_coh_scatter_for_type(self, n_sn, seed):
        raise ValueError(
            "Default scatterting for SNII not implemented yet, please provide SigM"
        )


class SNIcGen(CCGen):
    """SNIc class. Inherit from CCGen."""

    _object_type = "SNIc"
    _available_models = ut.Templatelist_fromsncosmo("snic") + CCGen._available_models
    _sn_lumfunc = {
        "M0": {"li11_gaussian": -16.75, "li11_skewed": -17.51},
        "coh_sct": {"li11_gaussian": [0.97, 0.97], "li11_skewed": [1.24, 1.22]},
    }

    _sn_fraction = {
        "shivers17": 0.075088,
        "ztf20": 0.110357,
    }
    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)",
    }


class SNIbGen(CCGen):
    """SNIb class. Inherit from CCGen."""

    _object_type = "SNIb"
    _available_models = ut.Templatelist_fromsncosmo("snib") + CCGen._available_models
    _sn_lumfunc = {
        "M0": {"li11_gaussian": -16.07, "li11_skewed": -17.71},
        "coh_sct": {"li11_gaussian": [1.34, 1.34], "li11_skewed": [2.11, 7.15]},
    }

    _sn_fraction = {
        "shivers17": 0.108224,
        "ztf20": 0.052551,
    }

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)",
    }


class SNIc_BLGen(CCGen):
    """SNIc_BL class. Inherit from CCGen."""

    _object_type = "SNIc_BL"
    _available_models = ut.Templatelist_fromsncosmo("snic-bl") + CCGen._available_models
    _sn_lumfunc = {
        "M0": {"li11_gaussian": -16.79, "li11_skewed": -17.74},
        "coh_sct": {"li11_gaussian": [0.95, 0.95], "li11_skewed": [1.35, 2.06]},
    }

    _sn_fraction = {
        "shivers17": 0.011248,
        "ztf20": 0.05421,
    }

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * {_sn_fraction['ztf20']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z: 9.10e-5 * {_sn_fraction['shivers17']} * ({{h}}/0.70)**3 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)",
    }


class SNIapeculiarGen(BaseGen):
    """SNIa_peculiar class.

     Models form platicc challenge ask Rick
     need a directory to store model

     Parameters
     ----------
    same as TimeSeriesGen class"""

    def _init_sources_list(self):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.
        """

        sources = self._sed_models
        return sources

    def gen_par(self, n_obj, basic_par, seed=None):

        params = {"sed_path": self._sed_path}

        if self._object_type.lower() == "sniax":
            rv, e_dust = self._gen_dust_par(n_obj, seed)

            params["E_dust"] = e_dust
            params["RV"] = rv

        return params

    def _add_print(self):
        str = ""
        return str

    def _update_header(self):
        header = {}
        header["M0_band"] = "bessell_v"
        return header


class SNIaxGen(SNIapeculiarGen):
    """SNIaxclass.

     Models form platicc challenge ask Rick
     need a directory to store model

     Parameters
     ----------
    same as TimeSeriesGen class"""

    _object_type = "SNIax"
    _available_models = "plasticc"
    _sed_models, _sed_path = plm.get_sed_listname("sniax")
    _available_rates = ["ptf19", "ptf19_pw"]

    _sn_fraction = {
        "plasticc": 0.24,
    }

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z:  2.43e-5 * {_sn_fraction['plasticc']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z:  2.43e-5 * {_sn_fraction['plasticc']} * ({{h}}/0.70)**3  *((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)",
    }

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
    _available_models = "plasticc"
    _sed_models, _sed_path = plm.get_sed_listname("snia91bg")
    _available_rates = ["ptf19", "ptf19_pw"]

    _sn_fraction = {
        "plasticc": 0.12,
    }

    _available_rates = {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z:  2.43e-5 * {_sn_fraction['plasticc']} * ({{h}}/0.70)**3",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19_pw": f"lambda z:  2.43e-5 * {_sn_fraction['plasticc']} * ({{h}}/0.70)**3  * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6)",
    }
