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
from . import utils as ut
from . import nb_fun as nbf


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

    _base_keys = ['expMJD',
                  'filter',
                  'fieldID',
                  'fieldRA',
                  'fieldDec']

    def __init__(self, survey_config):
        """Initialize SurveyObs class."""
        self._config = survey_config
        self._obs_table, self._start_end_days = self._init_data()
        self.fields = self._init_fields()

    def _init_fields(self):
        """Create a dictionnary with fieldID and coord.

        Returns
        -------
        dict
            fieldID : {'ra' : fieldRA, 'dec': fieldDec}.

        """
        # Create fields dic
        field_list = self.obs_table['fieldID'].unique()
        dic = {}
        for f in field_list:
            idx = nbf.find_first(f, self.obs_table['fieldID'].values)
            dic[f] = {'ra': self.obs_table['fieldRA'][idx],
                      'dec': self.obs_table['fieldDec'][idx]}

        # Check field shape
        if 'field_map' in self.config:
            field_map = self.config['field_map']
        else:
            field_map = 'rectangle'

        return SurveyFields(dic,
                            self.config['ra_size'],
                            self.config['dec_size'],
                            field_map)

    def print_config(self):
        print(f"SURVEY FILE : {self.config['survey_file']}")

        print("First day in survey_file : "
              f"{self.start_end_days[0].mjd:.2f} MJD / {self.start_end_days[0].iso}\n"
              "Last day in survey_file : "
              f"{self.start_end_days[1].mjd:.2f} MJD / {self.start_end_days[1].iso}")

        print(f"Survey effective duration is {self.duration:.2f} days")

        if 'survey_cut' in self.config:
            for k, v in self.config['survey_cut'].items():
                conditions_str = ''
                for cond in v:
                    conditions_str += str(cond) + ' OR '
                conditions_str = conditions_str[:-4]
                print(f'Select {k}: ' + conditions_str)
        else:
            print('No cut on survey file.')

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

        if ('fake_skynoise' not in self.config
           or self.config['fake_skynoise'][1].lower() == 'add'):
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

    def _extract_from_csv(self, keys):
        """Extract the observations table from csv file.

        Returns
        -------
        pandas.DataFrame
            The observations table.
        """
        obs_dic = pd.read_csv(self.config['survey_file'])

        if 'key_dic' in self.config:
            obs_dic.rename(columns=self.config['key_dic'],
                           inplace=True)

        for k in keys:
            if k not in obs_dic.keys().to_list():
                raise KeyError(f'{k} is needed in csv file')

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
        elif ext == '.csv':
            obs_dic = self._extract_from_csv(keys)
        else:
            raise ValueError('Accepted formats are .db and .csv')

        # avoid crash on errors by removing errors <= 0
        if ('fake_skynoise' not in self.config
           or self.config['fake_skynoise'][1].lower() == 'add'):
            obs_dic.query(f"{self.config['noise_key'][0]} > 0", inplace=True)

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

        # Reset index of the pandas DataFrame
        obs_dic.reset_index(drop=True, inplace=True)
        minMJDinObs = obs_dic['expMJD'].min()
        maxMJDinObs = obs_dic['expMJD'].max()

        # Effective start and end days
        start_day = ut.init_astropy_time(minMJDinObs)
        end_day = ut.init_astropy_time(maxMJDinObs)
        return obs_dic, (start_day, end_day)

    def epochs_selection(self, coords, model_t_range):
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
        Obj_ra, Obj_dec = coords

        epochs_selec = ((self.obs_table.expMJD > model_t_range[0])
                        & (self.obs_table.expMJD < model_t_range[1])).to_numpy()

        is_obs = epochs_selec.any()

        if is_obs:
            # Select the observed fields
            selec_fields_ID = self.obs_table['fieldID'][epochs_selec].unique()

            # Create a dic[fields] = obs_subfield
            dic_map = self.fields.is_in_field(Obj_ra, Obj_dec, selec_fields_ID)

            # Update the epochs_selec mask and check if there is some observations
            is_obs, epochs_selec = nbf.map_obs_fields(
                                                epochs_selec,
                                                self.obs_table['fieldID'][epochs_selec].to_numpy(),
                                                dic_map)

        if is_obs and 'sub_field' in self.config:
            is_obs, epochs_selec = nbf.map_obs_subfields(
                                epochs_selec,
                                self.obs_table['fieldID'][epochs_selec].to_numpy(),
                                self.obs_table[self.config['sub_field']][epochs_selec].to_numpy(),
                                dic_map)
        if is_obs:
            return self._make_obs_table(epochs_selec)
        return None

    def _make_obs_table(self, epochs_selec):
        """Create the astropy table from selection bool array.

        Parameters
        ----------
        epochs_selec : numpy.ndarray(boolean)
            A boolean array that define the observation selection.

        Returns
        -------
        astropy.Table
            The observations table that correspond to the selection.

        """
        obs_selec = self.obs_table.iloc[epochs_selec]
        # Change band name to correpond with sncosmo bands
        if self.band_dic is not None:
            band = obs_selec['filter'].map(self.band_dic).to_numpy(dtype='str')
        else:
            band = obs_selec['filter'].astype('U27').to_numpy(dtype='str')

        # Zero point selection
        if self.zp[0] != 'zp_in_obs':
            zp = np.ones(np.sum(epochs_selec)) * self.zp[0]
        else:
            zp = obs_selec['zp']

        # Sig Zero point selection
        if self.zp[1] != 'sig_zp_in_obs':
            sig_zp = np.ones(np.sum(epochs_selec)) * self.zp[1]
        else:
            sig_zp = obs_selec['sig_zp']

        # PSF selection
        if self.sig_psf != 'psf_in_obs':
            sig_psf = np.ones(np.sum(epochs_selec)) * self.sig_psf
        else:
            sig_psf = obs_selec['FWHMeff'] / (2 * np.sqrt(2 * np.log(2)))

        # Gain
        if self.gain != 'gain_in_obs':
            gain = np.ones(np.sum(epochs_selec)) * self.gain
        else:
            gain = obs_selec['gain']

        # Skynoise selection
        if ('fake_skynoise' not in self.config
           or self.config['fake_skynoise'][1].lower() == 'add'):
            if self.config['noise_key'][1] == 'mlim5':
                # Convert maglim to flux noise (ADU)
                mlim5 = obs_selec[self.config['noise_key'][0]]
                skynoise = 10.**(0.4 * (zp - mlim5)) / 5
            elif self.config['noise_key'][1] == 'skysigADU':
                skynoise = obs_selec[self.config['noise_key'][0]].copy()
            else:
                raise ValueError('Noise type should be mlim5 or skysigADU')
            if 'fake_skynoise' in self.config:
                skynoise = np.sqrt(skynoise**2 + self.config['fake_skynoise'][0])
        elif self.config['fake_skynoise'][1].lower() == 'replace':
            skynoise = np.ones(np.sum(epochs_selec)) * self.config['fake_skynoise'][0]
        else:
            raise ValueError("fake_skynoise type should be 'add' or 'replace'")

        # Apply PSF
        skynoise[sig_psf > 0] *= np.sqrt(4 * np.pi * sig_psf[sig_psf > 0]**2)

        # Create obs table following sncosmo formalism
        obs = pd.DataFrame({'time': obs_selec['expMJD'],
                            'band': band,
                            'gain': gain,
                            'skynoise': skynoise,
                            'zp': zp,
                            'sig_zp': sig_zp,
                            'fieldID': obs_selec['fieldID']})

        obs['zpsys'] = 'ab'
        if 'sub_field' in self.config:
            obs[self.config['sub_field']] = obs_selec[self.config['sub_field']]

        if 'add_data' in self.config:
            for k in self.config['add_data']:
                if k not in obs.keys():
                    if self.obs_table[k].dtype == 'object':
                        obs[k] = obs_selec[k].astype('U27').to_numpy(dtype='str')
                    else:
                        obs[k] = obs_selec[k]
        return obs


class SurveyFields:
    """Fields properties object.

    Parameters
    ----------
    fields_dic : dict
        ID and coordinates of fields.
    ra_size : float
        The RA size of the field in deg.
    dec_size : float
        The DEC size of the field in deg.
    field_map : str
        The path of the field map or just a str.
    """

    def __init__(self, fields_dic, ra_size, dec_size, field_map):
        """Init SurveyObs class."""
        self._size = np.array([ra_size, dec_size])
        self._dic = fields_dic
        self._sub_field_map = None
        self._init_fields_map(field_map)

    @property
    def size(self):
        """Get field size (ra,dec) in radians."""
        return np.radians(self._size)

    def read_sub_field_map(self, field_map):
        file = open(field_map)
        # Header symbol
        dic_symbol = {}
        nbr_id = -2
        lines = file.readlines()
        for i, l in enumerate(lines):
            if l[0] == '%':
                key_val = l[1:].strip().split(':')
                dic_symbol[key_val[0]] = {'nbr': nbr_id}
                dic_symbol[key_val[0]]['size'] = np.radians(float(key_val[2]))
                dic_symbol[key_val[0]]['type'] = key_val[1].lower()
                if key_val[1].lower() not in ['ra', 'dec']:
                    raise ValueError('Espacement type is ra or dec')
                nbr_id -= 1
            else:
                break

        # Compute void region
        # For the moment only work with regular grid
        subfield_map = [string.strip().split(':') for string in lines[i:] if string != '\n']
        used_ra = len(subfield_map[0])
        used_dec = len(subfield_map)
        ra_space = 0
        for k in dic_symbol.keys():
            if dic_symbol[k]['type'] == 'ra':
                ra_space += subfield_map[0].count(k) * dic_symbol[k]['size']
                used_ra -= subfield_map[0].count(k)
        dec_space = 0
        for lines in subfield_map:
            if lines[0] in dic_symbol.keys() and dic_symbol[lines[0]]['type'] == 'dec':
                dec_space += dic_symbol[lines[0]]['size']
                used_dec -= 1

        subfield_ra_size = (self.size[0] - ra_space) / used_ra
        subfield_dec_size = (self.size[1] - dec_space) / used_dec

        # Compute all ccd corner
        corner_dic = {}
        dec_metric = self.size[1] / 2
        for i, l in enumerate(subfield_map):
            if l[0] in dic_symbol and dic_symbol[l[0]]['type'] == 'dec':
                dec_metric -= dic_symbol[l[0]]['size']
            else:
                ra_metric = - self.size[0] / 2
                for j, elmt in enumerate(l):
                    if elmt in dic_symbol.keys() and dic_symbol[elmt]['type'] == 'ra':
                        ra_metric += dic_symbol[elmt]['size']
                    elif int(elmt) == -1:
                        ra_metric += subfield_ra_size
                    else:
                        corner_dic[int(elmt)] = np.array([
                                    [ra_metric, dec_metric],
                                    [ra_metric + subfield_ra_size, dec_metric],
                                    [ra_metric + subfield_ra_size, dec_metric - subfield_dec_size],
                                    [ra_metric, dec_metric - subfield_dec_size]])
                        ra_metric += subfield_ra_size
                dec_metric -= subfield_dec_size
        self.dic_sfld_file = dic_symbol
        return corner_dic

    def _init_fields_map(self, field_map):
        """Init the subfield map parameters..

        Parameters
        ----------
        field_map : dict
            ID: coordinates dict.

        Returns
        -------
        None
            Just set some attributes.

        """
        if field_map == 'rectangle':
            # Condition <=> always obs
            # Not good implemented
            self._sub_fields_corners = {0: np.array([[-self.size[0] / 2, self.size[1] / 2],
                                                     [self.size[0] / 2, self.size[1] / 2],
                                                     [self.size[0] / 2, -self.size[1] / 2],
                                                     [-self.size[0] / 2, -self.size[1] / 2]])}
        else:
            self._sub_fields_corners = self.read_sub_field_map(field_map)

    def is_in_field(self, SN_ra, SN_dec, fields_pre_selec=None):
        """Check if a SN is in a field and return the coordinates in the field frame.

        Parameters
        ----------
        SN_ra : float
            SN RA in radians.
        SN_dec : float
            SN DEC in radians.
        fields_pre_selec : numpy.array(int), opt
            A list of pre selected fields ID.

        Returns
        -------
        numba.Dict(int:bool), numba.Dict(int:numpy.array(float))
            The dictionnaries of boolena selection of obs fields and coordinates in observed fields.

        """
        if fields_pre_selec is not None:
            ra_fields, dec_fields = np.vectorize(
                                        lambda x: (self._dic.get(x)['ra'],
                                                   self._dic.get(x)['dec']))(fields_pre_selec)
        else:
            ra_fields = [self._dic[k]['ra'] for k in self._dic]
            dec_fields = [self._dic[k]['dec'] for k in self._dic]
            fields_pre_selec = [k for k in self._dic]

        # Compute the coord of the SN in the rest frame of each field
        obsfield_map = nbf.is_in_field(SN_ra,
                                       SN_dec,
                                       ra_fields,
                                       dec_fields,
                                       fields_pre_selec,
                                       np.array(list(self._sub_fields_corners)),
                                       np.array(list(self._sub_fields_corners.values())))
        return obsfield_map

    def show_map(self):
        """Plot a representation of subfields."""
        fig, ax = plt.subplots()
        for k, corners in self._sub_fields_corners.items():
            corners_deg = np.degrees(corners)
            p = Polygon(corners_deg, color='r', fill=False)
            ax.add_patch(p)
            x_text = 0.5 * (corners_deg[0][0] + corners_deg[1][0])
            y_text = 0.5 * (corners_deg[0][1] + corners_deg[3][1])
            ax.text(x_text, y_text, k, ha='center', va='center')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.set_xlim(-self._size[0] / 2 - 0.5, self._size[0] / 2 + 0.5)
        ax.set_ylim(-self._size[1] / 2 - 0.5, self._size[1] / 2 + 0.5)

        plt.show()


class SnHost:
    """Class containing the SN Host parameters.

    Parameters
    ----------
    config : str
        Configuration of host.
    z_range : list(float), opt
        The redshift range.
    """

    def __init__(self, config, z_range=None):
        """Initialize SnHost class."""
        self._z_range = z_range
        self._config = config
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

    def random_choice(self, n, rand_seed):
        """Randomly select hosts.

        Parameters
        ----------
        n : int
            Number of hosts to select.
        rand_gen : numpy.random.generator
            A numpy random generator.

        Returns
        -------
        pandas.dataframe
            Table with selected hosts properties.

        """
        rand_gen = np.random.default_rng(rand_seed)
        if self.config['distrib'].lower() == 'as_host':
            p = None
        elif self.config['distrib'].lower() == 'mass_weight':
            p = self.table['mass'] / self.table['mass'].sum()
        else:
            raise ValueError(f"{self.config['distrib']} is not an available option")
        idx = rand_gen.choice(self.table.index, p=p, size=n)
        return self.table.loc[idx]
