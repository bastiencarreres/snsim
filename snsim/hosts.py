"""This module contains the class which are used in the simulation."""

import warnings
import os
import numpy as np
from astropy.io import fits
import pandas as pd
import geopandas as gpd
from . import utils as ut


class SnHosts:
    """Class containing the SN Host parameters.

    Parameters
    ----------
    config : str
        Configuration of host.

        | config
        | ├── host_file, 'PATH/TO/HOSTFILE'
        | ├── distrib, str, Optional, default = 'rate', options given by self._dist_options
        | ├── reweight_vol, bool, Optional, default = False, reweight input host distrib to volumetric
        | └── key_dic: {'column_name': 'new_column_name', etc}, Optional, only use to change columns names

    z_range : list(float), opt
        The redshift range.
    """

    _dist_options = ["rate", "random", "mass", "mass_sfr", "sfr"]

    def __init__(self, config, z_range=None, geometry=None):
        """Initialize SnHost class."""
        self._config = config
        self._geometry = geometry
        self._z_range, self._df = self._read_host_file(z_range)
        self._max_dz = None

        # Default parameter
        if "distrib" not in self.config:
            self._config["distrib"] = "rate"
        elif self.config["distrib"].lower() not in self._dist_options:
            raise ValueError(
                f"{self.config['distrib']} is not an available option,"
                f"distributions are {self._dist_options}"
            )
        if "reweight_vol" not in self.config:
            self._config["reweight_vol"] = False
        if "host_noise" not in self.config:
            self._config["host_noise"] = False


    @property
    def config(self):
        """Get the configuration dic of host."""
        return self._config

    @property
    def max_dz(self):
        """Get the maximum redshift gap."""
        if self._max_dz is None:
            redshift_copy = np.sort(np.copy(self.df["zcos"]))
            diff = redshift_copy[1:] - redshift_copy[:-1]
            self._max_dz = np.max(diff)
        return self._max_dz

    @property
    def df(self):
        """Get pandas Dataframe of host."""
        return self._df

    def _read_host_file(self, z_range):
        """Extract host from host file.

        Returns
        -------
        pandas.Dataframe
            pandas Dataframe containing host.

        """
        ext = os.path.splitext(self.config["host_file"])[-1]

        if ext == ".fits":
            with fits.open(self.config["host_file"]) as hostf:
                host_list = pd.DataFrame.from_records(hostf[1].data[:])
        elif ext == ".csv":
            host_list = pd.read_csv(self.config["host_file"])
        elif ext == ".parquet":
            host_list = pd.read_parquet(self.config["host_file"])
        else:
            raise ValueError("Support .csv, .fits or .parquet files")

        if "key_dic" in self.config:
            key_dic = self.config["key_dic"]
        else:
            key_dic = {}

        #host_list = host_list.astype("float64")
        host_list.rename(columns=key_dic, inplace=True)
        host_list["ra"] += 2 * np.pi * (host_list["ra"] < 0)
        if z_range is not None:
            z_min, z_max = z_range
            if z_max > host_list["zcos"].max() or z_min < host_list["zcos"].min():
                warnings.warn(
                    "Simulation redshift range does not match host file redshift range",
                    UserWarning,
                )
            host_list.query(f"zcos >= {z_min} & zcos <= {z_max}", inplace=True)
        else:
            # By default give z range as host z range
            z_range = host_list.zcos.min(), host_list.zcos.max()
        if self._geometry is not None:
            ra_min, dec_min, ra_max, dec_max = self._geometry.bounds
            host_list.query(
                f"{ra_min} <= ra <= {ra_max} & {dec_min} <= dec <= {dec_max}",
                inplace=True,
            )

        host_list.reset_index(drop=True, inplace=True)
        return z_range, host_list

    def _reweight_volumetric(self, cosmology):
        count, zedges = np.histogram(self.df["zcos"], bins='rice')
        zcenter = (zedges[:-1] + zedges[1:]) * 0.5
        dV = cosmology.comoving_volume(zedges[1:]).value - cosmology.comoving_volume(zedges[:-1]).value
        count = count / np.sum(count)
        weights = np.interp(self.df["zcos"], zcenter, dV / count)
        return weights
        
    def compute_weights(self, rate=None, sn_type=None, cosmology=None):
        """Compute the weights for random choice.

        Parameters
        ----------
        rate : function, optional
            rate function of z, by default None.

        Returns
        -------
        numpy.ndarray(float)
            weigths for the random draw.
        """

        # Weights options
        if self.config["distrib"].lower() == "random":
            weights = None
        elif rate is not None:
            # Take into account rate is divide by (1 + z)
            weights_rate = rate(self.df["zcos"]) / (1 + self.df["zcos"])
            
            if self.config["distrib"].lower() == "rate":
                weights = weights_rate
            elif self.config["distrib"].lower() == "mass":
                # compute mass weight
                weights_mass = ut.compute_weight_mass_for_type(
                    mass=self.df["sm"], sn_type=sn_type, cosmology=cosmology
                )
                weights = weights_rate * weights_mass
            elif self.config["distrib"].lower() == "sfr":
                # compute SFR weight
                weights_SFR = ut.compute_weight_SFR_for_type(
                    SFR=self.df["sfr"], sn_type=sn_type, cosmology=cosmology
                )
                weights = weights_rate * weights_SFR
            elif self.config["distrib"].lower() == "mass_sfr":
                # compute SFR and mass weight
                weights_mass_sfr = ut.compute_weight_mass_sfr_for_type(
                    mass=self.df["sm"],sfr=self.df["sfr"], sn_type=sn_type, cosmology=cosmology
                )
                weights = weights_rate * weights_mass_sfr
            # Normalize
            weights /= weights.sum()
        else:
            raise ValueError("rate should be set to use host distribution")

        # Reweight currents distribution if asked
        if self.config["reweight_vol"]:
            vol_weights = self._reweight_volumetric(cosmology)
            if weights is not None:
                weights *= vol_weights
            else:
                weights = vol_weights

        # Normalize
        if weights is not None:
            weights /= weights.sum()
        return weights

        #TO DO: maybe generalize the distribution and add SFH dependence

    def random_choice(self, n, seed=None, rate=None, sn_type=None, cosmology=None):
        """Randomly select hosts.

        Parameters
        ----------
        n : int
            Number of hosts to select.
        seed : int, opt
            Random seed.

        Returns
        -------
        pandas.Dataframe
            Dataframe with selected hosts properties.

        """
        rand_gen = np.random.default_rng(seed)

        weights = self.compute_weights(rate=rate, sn_type=sn_type, cosmology=cosmology)

        if self._geometry is None:
            idx = rand_gen.choice(self.df.index, p=weights, size=n)
        else:
            idx = []
            n_to_sim = n
            while len(idx) < n:
                idx_tmp = np.random.choice(self.df.index, p=weights, size=n_to_sim)
                multipoint = gpd.points_from_xy(
                    self.df.loc[idx_tmp]["ra"], self.df.loc[idx_tmp]["dec"]
                )
                idx.extend(idx_tmp[multipoint.intersects(self._geometry)])
                n_to_sim = n - len(idx)

        return self.df.loc[idx]


def model_host_noise(sim_par, obs):
    """Compute noise coming from host galaxy in photoelectron units (approximation)"""

    mag_host = np.array([sim_par["host_mag_" + b] for b in obs['band']])
    flux_host = 10.**(0.4 * (obs['zp'] - mag_host))
    
    sersic_dic = {key: val for key, val in sim_par.items() if key.startswith('host_sersic')}
    n_sersic = len([k for k in sersic_dic.keys() if k.startswith('host_sersic_a')])    
    print(n_sersic)
    if n_sersic == 1:
        for k in sersic_dic.keys():
            if k[-1] != '0':
                sersic_dic[k+'0'] = sersic_dic.pop(k)
        sersic_dic['host_sersic_w0'] = 1
    else:
        # TODO: Not sure if it has sense for now to have multiple sersic profiles since we just approx F / area
        if np.sum([sersic_dic[f'host_sersic_w{i}'] for i in range(n_sersic)]) != 1:
            raise ValueError('Sersic w sum should be equal to 1')

    # compute mean  surface brightness of the galaxy
    surf_bright = np.sum([flux_host * sersic_dic[f'host_sersic_w{i}']  / (np.pi * sersic_dic[f'host_sersic_a{i}'] * sersic_dic[f'host_sersic_b{i}']) 
                        for i in range(n_sersic)])
            
    # Compute galaxy flux at SN position by applying PSF
    fpsf = surf_bright * np.pi * obs['fwhm_psf']**2 / (2 * np.log(2)) # = 4 * pi * SIG_psf

    return np.sqrt(np.abs(fpsf) / obs['gain'])