"""This module contains the class which are used in the simulation."""

import warnings
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import geopandas as gpd
from shapely import ops as shp_ops
import dask.dataframe as daskdf
from . import utils as ut
from . import geo_utils as geo_ut
from . import io_utils as io_ut
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
    | ├── survey_cut, CUT ON OBS FILE -> dict, opt
    | ├── add_data, LIST OF KEY TO ADD METADATA -> list(str), opt
    | ├── field_map, PATH TO SUBFIELD MAP FILE -> str, opt
    | └── sub_field, SUBFIELD KEY -> str, opt
    """

    # -- Basic keys needed in survey file (+ noise)
    _base_keys = ["expMJD", "filter", "fieldID", "fieldRA", "fieldDec"]

    def __init__(self, survey_config):
        """Initialize SurveyObs class."""

        self._config = survey_config

        # -- Init obs table
        self._obs_table, self._start_end_days = self._init_data()

        # -- Init fields
        if "field_map" in self.config:
            field_map = self.config["field_map"]
        else:
            field_map = "rectangle"

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
        restfield_corners = self._init_fields_map("rectangle")

        f_RA = np.array([minRA, maxRA, maxRA, minRA])
        f_Dec = np.array([maxDec, maxDec, minDec, minDec])

        sub_fields_corners = np.broadcast_to(
            restfield_corners[0], (4, *restfield_corners[0].shape)
        )

        corners = np.stack(
            [
                nbf.new_coord_on_fields(
                    sub_fields_corners[:, :, i, :], np.stack([f_RA, f_Dec])
                )
                for i in range(4)
            ],
            axis=1,
        )

        corners = geo_ut._format_corner(corners, f_RA)

        envelope = shp_ops.unary_union(
            [geo_ut._compute_polygon(corners[i]) for i in range(4)]
        ).envelope
        envelope_area = geo_ut._compute_area(envelope)
        return envelope, envelope_area

    def __str__(self):
        str = f"SURVEY FILE : {self.config['survey_file']}\n\n"

        str += (
            "First day in survey_file : "
            f"{self.start_end_days[0].mjd:.2f} MJD / {self.start_end_days[0].iso}\n"
            "Last day in survey_file : "
            f"{self.start_end_days[1].mjd:.2f} MJD / {self.start_end_days[1].iso}\n\n"
            f"Survey effective duration is {self.duration:.2f} days\n\n"
            f"Survey envelope area is {self._envelope_area * (180 / np.pi)**2:.2f} "
            "squared degrees "
            f"({self._envelope_area / (4 * np.pi) * 100:.1f} % of the sky)\n\n"
        )

        if "survey_cut" in self.config:
            for k, v in self.config["survey_cut"].items():
                conditions_str = ""
                for cond in v:
                    conditions_str += str(cond) + " OR "
                conditions_str = conditions_str[:-4]
                str += f"Select {k}: " + conditions_str + "\n"
        else:
            str += "No cut on survey file."
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
        min_mjd = obs_dic["expMJD"].min()
        max_mjd = obs_dic["expMJD"].max()

        if "start_day" in self.config:
            start_day = self.config["start_day"]
        else:
            start_day = min_mjd

        start_day = ut.init_astropy_time(start_day)

        if "end_day" in self.config:
            end_day = self.config["end_day"]
        elif "duration" in self.config:
            end_day = start_day.mjd + self.config["duration"]
        else:
            end_day = max_mjd

        end_day = ut.init_astropy_time(end_day)
        if end_day.mjd > max_mjd or start_day.mjd < min_mjd:
            warnings.warn(
                f"Starting day {start_day.mjd:.3f} MJD or"
                f"Ending day {end_day.mjd:.3f} MJD is outer of"
                f"the survey range : {min_mjd:.3f} - {max_mjd:.3f}",
                UserWarning,
            )

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

        keys += [self.config["noise_key"][0]]

        if "zp" not in self.config:
            keys += ["zp"]

        if "sig_zp" not in self.config:
            keys += ["sig_zp"]

        if "fwhm_psf" not in self.config:
            keys += ["fwhm_psf"]

        if "gain" not in self.config:
            keys += ["gain"]

        if "sub_field" in self.config:
            keys += [self.config["sub_field"]]

        if "add_data" in self.config:
            add_k = (k for k in self.config["add_data"] if k not in keys)
            keys += add_k
        return keys

    def _extract_from_file(self, ext, keys):
        """Extract the observations table from csv or parquet file.

        Returns
        -------
        pandas.DataFrame
            The observations table.
        """
        if ext == ".csv":
            obs_dic = pd.read_csv(self.config["survey_file"])
        elif ext == ".parquet":
            obs_dic = pd.read_parquet(self.config["survey_file"])

        # Optionnaly rename columns
        if "key_dic" in self.config:
            obs_dic.rename(columns=self.config["key_dic"], inplace=True)

        for k in keys:
            if k not in obs_dic.keys().to_list():
                raise KeyError(f"{k} is needed in obs file")

        if "survey_cut" in self.config:
            query = ""
            for cut_var in self.config["survey_cut"]:
                for cut in self.config["survey_cut"][cut_var]:
                    query += f"{cut_var}{cut} &"
            query = query[:-2]
            obs_dic.query(query, inplace=True)
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
        ext = os.path.splitext(self.config["survey_file"])[-1]

        # Init necessary keys
        keys = self._check_keys()

        if ext in [".csv", ".parquet"]:
            obs_dic = self._extract_from_file(ext, keys)
        else:
            raise ValueError("Accepted formats are .csv or .parquet")

        # Add noise key + avoid crash on errors by removing errors <= 0
        obs_dic.query(f"{self.config['noise_key'][0]} > 0", inplace=True)

        # Remove useless columns
        obs_dic = obs_dic[keys].copy()

        # Add zp, sig_zp, PSF and gain if needed
        if self.zp[0] != "zp_in_obs":
            obs_dic["zp"] = self.zp[0]

        if self.zp[1] != "sig_zp_in_obs":
            obs_dic["sig_zp"] = self.zp[1]

        if self.fwhm_psf != "psf_in_obs":
            obs_dic["fwhm_psf"] = self.fwhm_psf 

        if self.gain != "gain_in_obs":
            obs_dic["gain"] = self.gain

        # Keep only epochs in the survey time
        start_day_input, end_day_input = self._read_start_end_days(obs_dic)

        minMJDinObs = obs_dic["expMJD"].min()
        maxMJDinObs = obs_dic["expMJD"].max()

        if start_day_input.mjd < minMJDinObs:
            raise ValueError("start_day before first day in survey file")
        elif end_day_input.mjd > maxMJDinObs:
            raise ValueError("end_day after last day in survey file")

        obs_dic.query(
            f"expMJD >= {start_day_input.mjd} & expMJD <= {end_day_input.mjd}",
            inplace=True,
        )

        if obs_dic.size == 0:
            raise RuntimeError(
                "No observation for the given survey start_day and duration."
            )

        if not obs_dic["expMJD"].is_monotonic_increasing:
            obs_dic.sort_values("expMJD", inplace=True)

        # Reset index of the pandas DataFrame
        obs_dic.reset_index(drop=True, inplace=True)
        minMJDinObs = obs_dic["expMJD"].min()
        maxMJDinObs = obs_dic["expMJD"].max()

        # Change band name to correpond with sncosmo bands
        if self.band_dic is not None:
            obs_dic["filter"] = obs_dic["filter"].map(self.band_dic).astype("string")
        else:
            obs_dic["filter"] = obs_dic["filter"].astype("string")

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
        if field_config == "rectangle":
            sub_fields_corners = {
                0: np.array(
                    [
                        [
                            [-self.field_size_rad[0] / 2, self.field_size_rad[1] / 2],
                            [self.field_size_rad[0] / 2, self.field_size_rad[1] / 2],
                            [self.field_size_rad[0] / 2, -self.field_size_rad[1] / 2],
                            [-self.field_size_rad[0] / 2, -self.field_size_rad[1] / 2],
                        ]
                    ]
                )
            }
        else:
            sub_fields_corners = io_ut._read_sub_field_map(
                self.field_size_rad, field_config
            )

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

        Notes
        -----
        Inspired from  https://github.com/MickaelRigault/ztffields :
            ztffields.projection.spatialjoin_radec_to_fields
        """
        # -- Compute max and min of table section
        minMJD = df.expMJD.min()
        maxMJD = df.expMJD.max()

        ObjPoints = ObjPoints[(maxMJD >= ObjPoints.min_t) & (ObjPoints.max_t >= minMJD)]

        # -- Map field and rcid corners to their coordinates
        if "sub_field" in config:
            field_corners = np.stack(
                df[config["sub_field"]].map(sub_fields_corners).values
            )
        else:
            field_corners = np.broadcast_to(
                sub_fields_corners[0], (len(df), *sub_fields_corners[0].shape)
            )

        corners = np.stack(
            [
                nbf.new_coord_on_fields(
                    field_corners[:, :, i, :],
                    np.array([df['fieldRA'].values, df['fieldDec'].values]),
                )
                for i in range(4)
            ],
            axis=1,
        )

        corners = geo_ut._format_corner(corners, df['fieldRA'].values)

        # -- Create shapely polygon
        fgeo = np.vectorize(lambda i: geo_ut._compute_polygon(corners[i]))

        GeoS = gpd.GeoDataFrame(data=df, geometry=fgeo(np.arange(df.shape[0])))

        join = ObjPoints.sjoin(GeoS, how="inner", predicate="intersects")

        join["phase"] = (join["expMJD"] - join["t0"]) / join["1_zobs"]

        return join.drop(
            columns=["geometry", "index_right", "min_t", "max_t", "1_zobs", "t0"]
        )

    def get_observations(
        self,
        params,
        phase_cut=None,
        nep_cut=None,
        IDmin=0,
        use_dask=False,
        npartitions=None,
    ):
        """Give the epochs of observations of a given SN.

        Parameters
        ----------
        ra : numpy.ndarray(float) or float
            Obj ra coord [rad].
        dec : numpy.ndarray(float) or float
            Obj dec coord [rad].
        t0 : numpy.ndarray(float) or float
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
        ObjPoints = gpd.GeoDataFrame(
            data=params[["t0", "min_t", "max_t", "1_zobs"]],
            geometry=gpd.points_from_xy(params["ra"], params["dec"]),
            index=params.index,
        )

        if use_dask:
            if npartitions is None:
                # -- Arbitrary should be change
                npartitions = len(self.obs_table) // 10
            ddf = daskdf.from_pandas(self.obs_table, npartitions=npartitions)
            meta = daskdf.utils.make_meta(
                {**{k: t for k, t in zip(ddf.columns, ddf.dtypes)}, "phase": "float64"}
            )
            ObsObj = ddf.map_partitions(
                self._match_radec_to_obs,
                ObjPoints,
                self.config,
                self._sub_field_corners,
                align_dataframes=False,
                meta=meta,
            ).compute()
        else:
            ObsObj = self._match_radec_to_obs(
                self.obs_table, ObjPoints, self.config, self._sub_field_corners
            )
        # -- Phase cut
        if phase_cut is not None:
            ObsObj = ObsObj[
                (ObsObj['phase'] >= phase_cut[0]) & (ObsObj['phase'] <= phase_cut[1])
            ]

        if nep_cut is not None:
            for cut in nep_cut:
                test = (ObsObj['phase'] > cut[1]) & (ObsObj['phase'] < cut[2])
                if cut[3] != "any":
                    test &= ObsObj["filter"] == cut[3]
                test = test.groupby(level=0).sum() >= int(cut[0])

                ObsObj = ObsObj[ObsObj.index.map(test)]

        params = params.loc[ObsObj.index.unique()]

        # -- Reset index
        new_idx = {k: IDmin + i for i, k in enumerate(ObsObj.index.unique())}
        ObsObj["ID"] = ObsObj.index.map(new_idx)
        params["ID"] = params.index.map(new_idx)

        ObsObj.drop(columns="phase", inplace=True)
        ObsObj.set_index("ID", drop=True, inplace=True)
        params.set_index("ID", drop=True, inplace=True)

        # -- Sort the results
        ObsObj.sort_values(["ID", "expMJD"], inplace=True)
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
        Obs.rename(columns={"expMJD": "time", "filter": "band"}, inplace=True)
        Obs.drop(labels=["fieldRA", "fieldDec"], axis=1, inplace=True)

        # Skynoise selection
        if self.config["noise_key"][1] == "mlim5":
            # Convert maglim to flux noise (ADU)
            Obs["skynoise"] = 10.0 ** (0.4 * (Obs['zp'] - Obs[self.config["noise_key"][0]])) / 5
        elif self.config["noise_key"][1] == "skysigADU":
            Obs["skynoise"] = Obs[self.config["noise_key"][0]]
        else:
            raise ValueError("Noise type should be mlim5 or skysigADU")

        # Add CCD noise
        if "ccd_noise" in self.config:
            Obs["skynoise"] = np.sqrt(Obs["skynoise"]**2 + self.config["ccd_noise"]**2)

        # Apply PSF
        if self.fwhm_psf != 0:
            Obs["skynoise"] *= np.sqrt(4 * np.pi) * Obs["fwhm_psf"] / (2 * np.sqrt(2 * np.log(2)))
        
        

        # Magnitude system
        Obs["zpsys"] = "ab"

        return Obs

    def show_map(self, ax=None):
        """Plot a representation of subfields."""
        if ax is None:
            fig, ax = plt.subplots()
        for k, corners in self._sub_field_corners.items():
            corners_deg = np.degrees(corners)
            polist = [Polygon(cd, color="r", fill=False) for cd in corners_deg]
            for p in polist:
                ax.add_patch(p)
                x_text = 0.5 * (p.xy[0][0] + p.xy[1][0])
                y_text = 0.5 * (p.xy[0][1] + p.xy[3][1])
                ax.text(x_text, y_text, k, ha="center", va="center")
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.set_xlim(-self.config["ra_size"] / 2 - 0.5, self.config["ra_size"] / 2 + 0.5)
        ax.set_ylim(
            -self.config["dec_size"] / 2 - 0.5, self.config["dec_size"] / 2 + 0.5
        )
        ax.set_aspect("equal")
        if ax is None:
            plt.show()

    @property
    def config(self):
        """Survey configuration."""
        return self._config

    @property
    def band_dic(self):
        """Get the dic band_survey : band_sncosmo."""
        if "band_dic" in self.config:
            return self.config["band_dic"]
        return None

    @property
    def obs_table(self):
        """Table of the observations."""
        return self._obs_table

    @property
    def gain(self):
        """Get CCD gain in e-/ADU."""
        if "gain" in self._config:
            gain = self._config["gain"]
        else:
            gain = "gain_in_obs"
        return gain

    @property
    def zp(self):
        """Get zero point and it's uncertainty."""
        if "zp" in self._config:
            zp = self._config["zp"]
        else:
            zp = "zp_in_obs"
        if "sig_zp" in self._config:
            sig_zp = self._config["sig_zp"]
        else:
            sig_zp = "sig_zp_in_obs"
        return (zp, sig_zp)

    @property
    def fwhm_psf(self):
        """Get PSF width."""
        if "fwhm_psf" in self._config:
            fwhm_psf = self._config["fwhm_psf"]
        else:
            fwhm_psf = "psf_in_obs"
        return fwhm_psf

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
        return np.radians([self.config["ra_size"], self.config["dec_size"]])
