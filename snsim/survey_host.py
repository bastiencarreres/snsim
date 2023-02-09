"""This module contains the class which are used in the simulation."""

import sqlite3
import warnings
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from astropy.io import fits
import pandas as pd
import geopandas as gpd
from shapely import ops as shp_ops
import dask.dataframe as daskdf
from . import utils as ut
from . import io_utils as io_ut
from . import nb_fun as nbf
from .constants import C_LIGHT_KMS


class SurveyObs:
    """This class deals with the observations of the survey.

    Parameters
    ----------
    survey_config : dic
        It contains all the survey configuration.

      | survey_config
      | ├── survey_file PATH TO SURVEY FILE
      | ├── ra_size RA FIELD SIZE IN DEG -> float
      | ├── dec_size DEC FIELD SIZE IN DEG -> float
      | ├── gain CCD GAIN e-/ADU -> float
      | ├── start_day STARTING DAY -> float or str, opt
      | ├── end_day ENDING DAY -> float or str, opt
      | ├── duration SURVEY DURATION -> float, opt
      | ├── zp FIXED ZEROPOINT -> float, opt
      | ├── survey_cut, CUT ON DB FILE -> dict, opt
      | ├── add_data, LIST OF KEY TO ADD METADATA -> list(str), opt
      | ├── field_map, PATH TO SUBFIELD MAP FILE -> str, opt
      | └── sub_field, SUBFIELD KEY -> str, opt
    """

    # -- Basic keys needed in survey file (+ noise)
    _base_keys = ['expMJD',
                  'filter',
                  'fieldID',
                  'fieldRA',
                  'fieldDec']

    def __init__(self, survey_config):
        """Initialize SurveyObs class."""

        self._config = survey_config

        # -- Init obs table
        self._obs_table, self._start_end_days = self._init_data()

        # -- Init fields
        if 'field_map' in self.config:
            field_map = self.config['field_map']
        else:
            field_map = 'rectangle'

        self._sub_field_corners = self._init_fields_map(field_map)         
        self._envelope, self._envelope_area = self._compute_envelope()
  
    def _compute_envelope(self):
        """Compute envelope of survey geometry and it's area.

        Returns
        -------
        shapely.Polygon, float
            envelope of the survey, it's area
        """
        # Compute min and max positons
        minDec = self.obs_table.fieldDec.min()
        maxDec = self.obs_table.fieldDec.max()
        minRA = self.obs_table.fieldRA.min()
        maxRA = self.obs_table.fieldRA.max()

        # Represent them as rectangle
        restfield_corners = self._init_fields_map('rectangle')

        f_RA = [minRA, maxRA, maxRA, minRA]
        f_Dec = [maxDec, maxDec, minDec, minDec]

        sub_fields_corners = np.broadcast_to(restfield_corners[0], (4, 4, 2))

        corners = {}
        for i in range(4):
            corners[i] = nbf.new_coord_on_fields(sub_fields_corners[:, i].T, 
                                                 [f_RA, f_Dec])
        corners = ut._format_corner(corners, f_RA)
        envelope = shp_ops.unary_union([ut._compute_polygon([[corners[i][0][j],
                                                              corners[i][1][j]] 
                                                            for i in range(4)]) 
                                        for j in range(4)]).envelope
        envelope_area = ut._compute_area(envelope)
        return envelope, envelope_area
        
    def __str__(self):
        str = f"SURVEY FILE : {self.config['survey_file']}\n\n"

        str += ("First day in survey_file : "
                f"{self.start_end_days[0].mjd:.2f} MJD / {self.start_end_days[0].iso}\n"
                "Last day in survey_file : "
                f"{self.start_end_days[1].mjd:.2f} MJD / {self.start_end_days[1].iso}\n\n"
                f"Survey effective duration is {self.duration:.2f} days\n\n"
                f"Survey envelope area is {self._envelope_area * (180 / np.pi)**2:.2f} "
                "squared degrees "
                f"({self._envelope_area / (4 * np.pi) * 100:.1f} % of the sky)\n\n")

        if 'survey_cut' in self.config:
            for k, v in self.config['survey_cut'].items():
                conditions_str = ''
                for cond in v:
                    conditions_str += str(cond) + ' OR '
                conditions_str = conditions_str[:-4]
                str += (f'Select {k}: ' + conditions_str + '\n')
        else:
            str += 'No cut on survey file.'
        return str

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
        min_mjd = obs_dic['expMJD'].min()
        max_mjd = obs_dic['expMJD'].max()
        if 'start_day' in self.config:
            start_day = self.config['start_day']
        else:
            start_day = min_mjd

        start_day = ut.init_astropy_time(start_day)

        if 'end_day' in self.config:
            end_day = self.config['end_day']
        elif 'duration' in self.config:
            end_day = start_day.mjd + self.config['duration']
        else:
            end_day = max_mjd

        end_day = ut.init_astropy_time(end_day)
        if end_day.mjd > max_mjd or start_day.mjd < min_mjd:
            warnings.warn(f'Starting day {start_day.mjd:.3f} MJD or'
                          f'Ending day {end_day.mjd:.3f} MJD is outer of'
                          f'the survey range : {min_mjd:.3f} - {max_mjd:.3f}',
                          UserWarning)

        if end_day.mjd < start_day.mjd:
            raise ValueError("The ending day is before the starting day !")
        return start_day, end_day

    def _check_keys(self):
        """Check which keys are needed.

        Returns
        -------
        list(str)
            All keys needed.

        """
        keys = copy.copy(self._base_keys)

        keys += [self.config['noise_key'][0]]

        if 'zp' not in self.config:
            keys += ['zp']

        if 'sig_zp' not in self.config:
            keys += ['sig_zp']

        if 'sig_psf' not in self.config:
            keys += ['FWHMeff']

        if 'gain' not in self.config:
            keys += ['gain']

        if 'sub_field' in self.config:
            keys += [self.config['sub_field']]

        if 'add_data' in self.config:
            add_k = (k for k in self.config['add_data'] if k not in keys)
            keys += add_k
        return keys

    def _extract_from_file(self, ext, keys):
        """Extract the observations table from csv or parquet file.

        Returns
        -------
        pandas.DataFrame
            The observations table.
        """
        if ext == '.csv':
            obs_dic = pd.read_csv(self.config['survey_file'])
        elif ext == '.parquet':
            obs_dic = pd.read_parquet(self.config['survey_file'])

        # Optionnaly rename columns
        if 'key_dic' in self.config:
            obs_dic.rename(columns=self.config['key_dic'],
                           inplace=True)

        for k in keys:
            if k not in obs_dic.keys().to_list():
                raise KeyError(f'{k} is needed in obs file')

        if 'survey_cut' in self.config:
            query = ''
            for cut_var in self.config['survey_cut']:
                for cut in self.config['survey_cut'][cut_var]:
                    query += f'{cut_var}{cut} &'
            query = query[:-2]
            obs_dic.query(query,
                          inplace=True)
        return obs_dic

    def _extract_from_db(self, keys):
        """Extract the observations table from SQL data base.

        Returns
        -------
        pandas.DataFrame
            The observations table.
        """
        con = sqlite3.connect(self.config['survey_file'])

        # Create the SQL query
        where = ''
        if 'survey_cut' in self.config:
            where = " WHERE "
            for cut_var in self.config['survey_cut']:
                where += "("
                for cut in self.config['survey_cut'][cut_var]:
                    cut_str = f"{cut}"
                    where += f"{cut_var}{cut_str} AND "
                where = where[:-4]
                where += ") AND "
            where = where[:-5]
        query = 'SELECT '
        for k in keys:
            query += k + ','
        query = query[:-1]
        query += ' FROM Summary' + where + ';'
        obs_dic = pd.read_sql_query(query, con)
        return obs_dic

    def _init_data(self):
        """Initialize observations table.

        Returns
        -------
        pandas.DataFrame
            The observations table.
        tuple(astropy.time.Time)
            The starting time and ending time of the survey.
        """
        # Extract extension
        ext = os.path.splitext(self.config['survey_file'])[-1]

        # Init necessary keys
        keys = self._check_keys()
        if ext == '.db':
            obs_dic = self._extract_from_db(keys)
        elif ext in ['.csv', '.parquet']:
            obs_dic = self._extract_from_file(ext, keys)
        else:
            raise ValueError('Accepted formats are .db, .csv or .parquet')

        # Add noise key + avoid crash on errors by removing errors <= 0
        obs_dic.query(f"{self.config['noise_key'][0]} > 0", inplace=True)

        # Remove useless columns
        obs_dic = obs_dic[keys].copy()

        # Add zp, sig_zp, PSF and gain if needed
        if self.zp[0] != 'zp_in_obs':
            obs_dic['zp'] = self.zp[0]

        if self.zp[1] != 'sig_zp_in_obs':
            obs_dic['sig_zp'] = self.zp[1]

        if self.sig_psf != 'psf_in_obs':
            obs_dic['sig_psf'] = self.sig_psf

        if self.gain != 'gain_in_obs':
            obs_dic['gain'] = self.gain

        # Keep only epochs in the survey time
        start_day_input, end_day_input = self._read_start_end_days(obs_dic)

        minMJDinObs = obs_dic['expMJD'].min()
        maxMJDinObs = obs_dic['expMJD'].max()

        if start_day_input.mjd < minMJDinObs:
            raise ValueError('start_day before first day in survey file')
        elif end_day_input.mjd > maxMJDinObs:
            raise ValueError('end_day after last day in survey file')

        obs_dic.query(f"expMJD >= {start_day_input.mjd} & expMJD <= {end_day_input.mjd}",
                      inplace=True)

        if obs_dic.size == 0:
            raise RuntimeError('No observation for the given survey start_day and duration.')

        if not obs_dic['expMJD'].is_monotonic_increasing:
            obs_dic.sort_values('expMJD', inplace=True)

        # Reset index of the pandas DataFrame
        obs_dic.reset_index(drop=True, inplace=True)
        minMJDinObs = obs_dic['expMJD'].min()
        maxMJDinObs = obs_dic['expMJD'].max()

        # Change band name to correpond with sncosmo bands
        if self.band_dic is not None:
            obs_dic['filter'] = obs_dic['filter'].map(self.band_dic).astype('string')
        else:
            obs_dic['filter'] = obs_dic['filter'].astype('string')

        # Effective start and end days
        start_day = ut.init_astropy_time(minMJDinObs)
        end_day = ut.init_astropy_time(maxMJDinObs)
        return obs_dic, (start_day, end_day)


    def _init_fields_map(self, field_config):
        """Init the subfield map parameters.

        Parameters
        ----------
        field_config : str 
            Shape or file that contains the sub-field description.

        Returns
        -------
        dict
            sub-field corners postion.

        """
        if field_config == 'rectangle':
            sub_fields_corners = {0: np.array([[-self.field_size_rad[0] / 2,  self.field_size_rad[1] / 2],
                                               [ self.field_size_rad[0] / 2,  self.field_size_rad[1] / 2],
                                               [ self.field_size_rad[0] / 2, -self.field_size_rad[1] / 2],
                                               [-self.field_size_rad[0] / 2, -self.field_size_rad[1] / 2]])}
        else:
            sub_fields_corners = io_ut._read_sub_field_map(self.field_size_rad, field_config)

        return sub_fields_corners

    @staticmethod
    def _match_radec_to_obs(df, ObjPoints, config, sub_fields_corners):
        """Return observation of ObjPoints.

        Parameters
        ----------
        df : pandas.DataFrame
            Observations.
        ObjPoints : geopandas.DataFrame
            Position of object.
        
        Return
        ------
        pandas.DataFrame
            Observations of objects.
        """        
        # -- Map field and rcid corners to their coordinates
        if 'sub_field' in config:
            field_corners = np.stack(df[config['sub_field']].map(sub_fields_corners).values)
        else:
            field_corners = np.broadcast_to(sub_fields_corners[0], (len(df), 4, 2))
        
        corner = {}
        for i in range(4):
            corner[i] = nbf.new_coord_on_fields(field_corners[:, i].T, 
                                                [df.fieldRA.values, df.fieldDec.values])
        
        corner = ut._format_corner(corner, df.fieldRA.values)

        # -- Create shapely polygon
        geometry = [ut._compute_polygon([[corner[i][0][j], corner[i][1][j]] for i in range(4)]) 
                                         for j in range(len(df))]
        
        GeoS = gpd.GeoDataFrame(data=df, 
                                geometry=geometry)
        
        join = ObjPoints.sjoin(GeoS, how="inner", predicate="intersects")

        return join.drop(columns=['geometry', 'index_right'])

    def get_observations(self, params, phase_cut=None, nep_cut=None, IDmin=0, 
                         use_dask=False, npartitions=None):
        """Give the epochs of observations of a given SN.

        Parameters
        ----------
        ra : numpy.ndarray(float) or float
            Obj ra coord [rad].
        dec : numpy.ndarray(float) or float
            Obj dec coord [rad].
        sim_t0 : numpy.ndarray(float) or float
            Obj sncosmo model peak time.
        MinT : numpy.ndarray(float) or float
            Obj sncosmo model mintime.
        MaxT : numpy.ndarray(float) or float
            Obj sncosmo model maxtime.
        nep_cut : list(list(int, float, float, str)), opt
            The cut [nep, mintime, maxtime, band].
        IDmin : int, opt
            ID of the first object.

        Returns
        -------
        pandas.DataFrame()
            pandas dataframe containing the observations.

        """
        params = params.copy()
        ObjPoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy(params['ra'], params['dec']),
                                     index=params.index)
        
        if use_dask:
            if npartitions is None: 
                # -- Arbitrary should be change
                npartitions = len(self.obs_table) // 10
            ddf = daskdf.from_pandas(self.obs_table, npartitions=npartitions)
            ObsObj = ddf.map_partitions(self._match_radec_to_obs,
                                        ObjPoints, self.config,
                                        self._sub_field_corners,
                                        align_dataframes=False,
                                        meta=ddf).compute()
            self.test = ObsObj
        else:
            ObsObj = self._match_radec_to_obs(self.obs_table, ObjPoints,
                                              self.config, self._sub_field_corners)
        # -- Phase mask
        _1_zobs_ = ((1 + params.zcos[ObsObj.index]) * (1 + params.vpec[ObsObj.index] / C_LIGHT_KMS))
        phase = (ObsObj.expMJD - params.sim_t0[ObsObj.index]) / _1_zobs_
            
        if phase_cut is not None:
            phase_mask = (phase >= phase_cut[0]) & (phase <= phase_cut[1])
            ObsObj = ObsObj[phase_mask]

        if nep_cut is not None:
            phase = phase[phase_mask]
            for cut in nep_cut:
                test = (phase > cut[1]) & (phase < cut[2])
                if cut[3] != 'any':
                    test &= ObsObj['filter'] == cut[3]
                test = test.groupby(level=0).sum() > int(cut[0])

                ObsObj = ObsObj[ObsObj.index.map(test)]
                phase = phase[phase.index.map(test)]

        params = params.loc[ObsObj.index.unique()]

        # -- Reset index
        new_idx = {k:IDmin + i for i, k in enumerate(ObsObj.index.unique())}
        ObsObj['ID'] = ObsObj.index.map(new_idx)
        params['ID'] = params.index.map(new_idx)
        ObsObj.set_index('ID', drop=True, inplace=True)
        params.set_index('ID', drop=True, inplace=True)

        ObsObj.sort_index(inplace=True)
        params.sort_index(inplace=True)
        if len(params) == 0:
            return None, None
        return self._make_obs_table(ObsObj), params

    def _make_obs_table(self, Obs):
        """Format observations of object.

        Parameters
        ----------
        Obs : pandas.DataFrame
            A boolean array that define the observation selection.

        Returns
        -------
        pandas.DataFrame
            The observations table that correspond to the selection.

        """
        Obs.rename(columns={'expMJD': 'time', 'filter': 'band'}, inplace=True)
        Obs.drop(labels=['fieldRA', 'fieldDec'], axis=1, inplace=True)

        # PSF selection
        if self.sig_psf == 'psf_in_obs':
            Obs['sig_psf'] = Obs['FWHMeff'] / (2 * np.sqrt(2 * np.log(2)))
            Obs.drop(columns='FWHMeff', inplace=True)

        # Skynoise selection
        if self.config['noise_key'][1] == 'mlim5':
            # Convert maglim to flux noise (ADU)
            mlim5 = Obs[self.config['noise_key'][0]]
            skynoise = 10.**(0.4 * (Obs.zp - mlim5)) / 5
        elif self.config['noise_key'][1] == 'skysigADU':
            skynoise = Obs[self.config['noise_key'][0]].copy()
        else:
            raise ValueError('Noise type should be mlim5 or skysigADU')

        # Apply PSF
        psf_mask = Obs.sig_psf > 0
        skynoise[psf_mask] *= np.sqrt(4 * np.pi * Obs['sig_psf'][psf_mask]**2)

        # Skynoise column
        Obs['skynoise'] = skynoise

        # Magnitude system
        Obs['zpsys'] = 'ab'

        return Obs

    def show_map(self, ax=None):
        """Plot a representation of subfields."""
        if ax is None:
            fig, ax = plt.subplots()
        for k, corners in self._sub_field_corners.items():
            corners_deg = np.degrees(corners)
            p = Polygon(corners_deg, color='r', fill=False)
            ax.add_patch(p)
            x_text = 0.5 * (corners_deg[0][0] + corners_deg[1][0])
            y_text = 0.5 * (corners_deg[0][1] + corners_deg[3][1])
            ax.text(x_text, y_text, k, ha='center', va='center')
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('Dec [deg]')
        ax.set_xlim(-self.config['ra_size'] / 2 - 0.5, 
                    self.config['ra_size']  / 2 + 0.5)
        ax.set_ylim(-self.config['dec_size']  / 2 - 0.5, 
                    self.config['dec_size'] / 2 + 0.5)
        ax.set_aspect('equal')
        if ax is None:
            plt.show()

    @property
    def config(self):
        """Survey configuration."""
        return self._config

    @property
    def band_dic(self):
        """Get the dic band_survey : band_sncosmo."""
        if 'band_dic' in self.config:
            return self.config['band_dic']
        return None

    @property
    def obs_table(self):
        """Table of the observations."""
        return self._obs_table

    @property
    def gain(self):
        """Get CCD gain in e-/ADU."""
        if 'gain' in self._config:
            gain = self._config['gain']
        else:
            gain = 'gain_in_obs'
        return gain

    @property
    def zp(self):
        """Get zero point and it's uncertainty."""
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
        """Get PSF width."""
        if 'sig_psf' in self._config:
            sig_psf = self._config['sig_psf']
        else:
            sig_psf = 'psf_in_obs'
        return sig_psf

    @property
    def duration(self):
        """Get the survey duration in days."""
        duration = self.start_end_days[1].mjd - self.start_end_days[0].mjd
        return duration

    @property
    def start_end_days(self):
        """Get the survey start and ending days."""
        return self._start_end_days[0], self._start_end_days[1]
    
    @property
    def field_size_rad(self):
        """Get field size ra, dec in radians."""
        return np.radians([self.config['ra_size'], self.config['dec_size']])


class SnHost:
    """Class containing the SN Host parameters.

    Parameters
    ----------
    config : str
        Configuration of host.
    z_range : list(float), opt
        The redshift range.
    """

    def __init__(self, config, z_range=None, geometry=None):
        """Initialize SnHost class."""
        self._z_range = z_range
        self._config = config
        self._geometry = geometry
        self._table = self._read_host_file()
        self._max_dz = None

        # Default parameter
        if 'distrib' not in self.config:
            self._config['distrib'] = 'as_sn'

    @property
    def config(self):
        """Get the configuration dic of host."""
        return self._config

    @property
    def max_dz(self):
        """Get the maximum redshift gap."""
        if self._max_dz is None:
            redshift_copy = np.sort(np.copy(self.table['redshift']))
            diff = redshift_copy[1:] - redshift_copy[:-1]
            self._max_dz = np.max(diff)
        return self._max_dz

    @property
    def table(self):
        """Get astropy Table of host."""
        return self._table

    def _read_host_file(self):
        """Extract host from host file.

        Returns
        -------
        astropy.Table
            astropy Table containing host.

        """
        ext = os.path.splitext(self.config['host_file'])[-1]

        if ext == '.fits':
            with fits.open(self.config['host_file']) as hostf:
                host_list = pd.DataFrame.from_records(hostf[1].data[:])
        elif ext == '.csv':
            host_list = pd.read_csv(self.config['host_file'])
        elif ext == '.parquet':
            host_list = pd.read_parquet(self.config['host_file'])
        else:
            raise ValueError('Support .csv, .fits or .parquet files')

        if 'key_dic' in self.config:
            key_dic = self.config['key_dic']
        else:
            key_dic = {}

        host_list = host_list.astype('float64')
        host_list.rename(columns=key_dic, inplace=True)
        ra_mask = host_list['ra'] < 0
        host_list['ra'][ra_mask] = host_list['ra'][ra_mask] + 2 * np.pi
        if self._z_range is not None:
            z_min, z_max = self._z_range
            if (z_max > host_list['redshift'].max()
               or z_min < host_list['redshift'].min()):
                warnings.warn('Simulation redshift range does not match host file redshift range',
                              UserWarning)
            host_list.query(f'redshift >= {z_min} & redshift <= {z_max}', inplace=True)
        if self._geometry is not None:
            ra_min, dec_min, ra_max, dec_max = self._geometry.bounds
            host_list.query(f'{ra_min} <= ra <= {ra_max} & {dec_min} <= dec <= {dec_max}',
                            inplace=True)

        host_list.reset_index(drop=True, inplace=True)
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

    def _normalize_distrib(self):
        count, egdes = np.histogram(self.table['redshift'], bins=1000,
                                    range=[self.table['redshift'].min(), self.table['redshift'].max()])
        count = count / np.sum(count)
        zcenter = (egdes[:-1] + egdes[1:]) * 0.5
        p = np.interp(self.table['redshift'], zcenter, count)
        p_inv = 1 / p
        p_inv /= np.sum(p_inv)
        return p_inv

    def random_choice(self, n, seed=None, z_dist=None):
        """Randomly select hosts.

        Parameters
        ----------
        n : int
            Number of hosts to select.
        seed : int, opt
            Random seed.

        Returns
        -------
        pandas.dataframe
            Table with selected hosts properties.

        """
        rand_gen = np.random.default_rng(seed)
        
        if self.config['distrib'].lower() == 'as_host':
            # Take into account rate is divide by (1 + z)
            choice_weights = 1 / (1 + self.table['redshift'])
            choice_weights /= choice_weights.sum()
        elif self.config['distrib'].lower() == 'mass_weight':
            choice_weights = self.table['mass'] / self.table['mass'].sum()
        elif self.config['distrib'].lower() == 'as_sn':
            norm = self._normalize_distrib()
            prob_z = np.gradient(z_dist.cdf, z_dist.x)
            Pz = np.interp(self.table['redshift'],  z_dist.x, prob_z)
            choice_weights = norm * Pz
            choice_weights /= choice_weights.sum()
        else:
            raise ValueError(f"{self.config['distrib']} is not an available option")

        if self._geometry is None:
            idx = rand_gen.choice(self.table.index, p=choice_weights, size=n)
        else:
            idx = []
            n_to_sim = n
            while len(idx) < n:
                idx_tmp = np.random.choice(self.table.index, p=choice_weights, size=n_to_sim)
                multipoint = gpd.points_from_xy(self.table.loc[idx_tmp]['ra'], 
                                                self.table.loc[idx_tmp]['dec'])
                idx.extend(idx_tmp[multipoint.intersects(self._geometry)])
                n_to_sim = n - len(idx)

        return self.table.loc[idx]
