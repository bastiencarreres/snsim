"""SimSample class used to store simulations."""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from . import utils as ut
from . import scatter as sct
from . import plot_utils as plot_ut
from . import dust_utils as dst_ut
from . import io_utils as io_ut

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class SimSample:
    """Class to store simulated SN sample.

    Parameters
    ----------
    sample_name : str
        Name of the sample.
    sim_lcs : list(astropy.Table)
        The simulated lightcurves.
    header : dict
        Simulation header.
    model_dir : str, opt
        The path to the simulation model files.
    file_path : str, opt
        Path of the sample.
    """

    def __init__(self, sample_name, sim_lcs, header, model_dir=None, dir_path=None):
        """Initialize SimSample class."""
        self._name = sample_name
        self._header = header
        self._sim_lcs = copy.copy(sim_lcs)
        self._model_dir = model_dir
        self._dir_path = dir_path

        self._sim_model = self._init_sim_model()

        self._fit_model = None
        self._fit_res = None

        self.modified_lcs = copy.copy(sim_lcs)

        self._bands = self.sim_lcs['band'].unique()

    @classmethod
    def fromDFlist(cls, sample_name, sim_lcs, header, model_dir=None, dir_path=None):
        """Initialize the class from a list of pandas.DataFrame.

        Parameters
        ----------
        cls : SimSample class
            The SimSample class.
        sim_file : str
            The file to load.
        model_dir : str, opt
            The directory of the configuration files of the sim model.
        sample_name : str
            Name of the simulation.
        sim_lcs : list(pandas.DataFrame)
            The sim lightcurves.
        header : dict
            Simulation header.
        model_dir : str, optional default is None
            Simulation model directory.
        dir_path : str, optional default is None
            Path to the output directory.

        Returns
        -------
        SimSample class object
            A SimSample class with the simulated lcs.

        """
        lcs = pd.concat(sim_lcs,
                        keys=(lc.attrs['ID'] for lc in sim_lcs),
                        names=['ID'])
        lcs.attrs = {lc.attrs['ID']: lc.attrs for lc in sim_lcs}
        return cls(sample_name, lcs, header, model_dir=model_dir,
                   dir_path=dir_path)

    @classmethod
    def fromFile(cls, sim_file, model_dir=None):
        """Initialize the class from a fits or pickle file.

        Parameters
        ----------
        cls : SimSample class
            The SimSample class.
        sim_file : str
            The file to load.
        model_dir : str, opt
            The directory of the configuration files of the sim model.

        Returns
        -------
        SimSample class object
            A SimSample class with the simulated lcs.

        """
        name, header, lcs = io_ut.read_sim_file(sim_file)

        return cls(name, lcs, header,
                   model_dir=model_dir, dir_path=os.path.dirname(sim_file) + '/')

    def _init_sim_model(self):
        """Initialize sim model.

        Returns
        -------
        sncosmo.Model
            sncosmo sim model.

        """
        model_name = self.header['model_name']

        sim_model = ut.init_sn_model(model_name, self._model_dir)

        if 'sct_mod' in self.header:
            sct.init_sn_sct_model(sim_model, self.header['sct_mod'])

        if 'mw_mod' in self.header:
            dst_ut.init_mw_dust(sim_model, {'model': self.header['mw_mod'],
                                            'rv': self.header['mw_rv']})
        return sim_model

    def _set_obj_effects_model(self, model, ID):
        """Set the model parameters of one obj.

        Parameters
        ----------
        model : sncosmo.Model
            The model to use.
        ID : int
            obj ID.

        Returns
        -------
        None
            Directly set model parameters.

        """
        for effect, eff_name in zip(model.effects,
                                    model.effect_names):
            for par in effect.param_names:
                pname = eff_name + par
                if pname in self.sim_lcs.attrs[ID]:
                    model.set(**{pname: self.sim_lcs.attrs[ID][pname]})
                elif pname in self.header:
                    model.set(**{pname: self.header[pname]})
                else:
                    raise AttributeError(f'{pname} not found')

    def get_obj_sim_model(self, obj_ID):
        """Get th sim model of one obj.

        Parameters
        ----------
        obj_ID : int
            ID of the obj.

        Returns
        -------
        sncosmo.Model
            sncosmo sim model of the obj.

        """
        simmod = copy.copy(self._sim_model)
        par = {'z': self.meta[obj_ID]['zobs'],
               't0': self.meta[obj_ID]['sim_t0']}

        if self.header['obj_type'].lower() == 'snia':
            par = {**par, **self._get_snia_simsncpar(obj_ID)}

        simmod.set(**par)
        self._set_obj_effects_model(simmod, obj_ID)

        return simmod

    def set_fit_model(self, model, model_dir=None, mw_dust=None):
        """Set the fit with a given SNCosmo Model.

        Parameters
        ----------
        model : sncosmo.models.Model or str
            A sncosmo Model or a model name for utils.init_sn_model(model, model_dir).
        model_dir : str, opt
            In case you want to set your model via utils.init_sn_model(model, model_dir).
        mw_dust : dict, opt
            Milky Wat dust model to apply.
          | mw_dust
          | ├── model : str (model name)
          | └── rv : float, opt
        Returns
        -------
        None
            Set the self._fit_model attribute.
        """
        if isinstance(model, type(ut.init_sn_model('salt2'))):
            self._fit_model = copy.copy(model)
        elif isinstance(model, str):
            self._fit_model = ut.init_sn_model(self.header['model_name'], model_dir)
        else:
            raise ValueError('Input can be a sncosmo model or a string')
        if mw_dust is not None:
            dst_ut.init_mw_dust(self._fit_model, mw_dust)

    def fit_lc(self, obj_ID=None, **kwargs):
        """Fit all or just one SN lightcurve(s).

        Parameters
        ----------
        ID : int, default is None
            The SN ID, if not specified all SN are fit.

        Returns
        -------
        None
            Directly modified the _fit_res attribute.

        Notes
        -----
        Use snc_fitter from utils

        """
        if self.fit_model is None:
            raise ValueError('Set fit model before launch fit')

        if self._fit_res is None:
            self._fit_res = {}

        fit_model = self.fit_model.__copy__()

        if fit_model.source.name[:4] == 'salt':
            fit_par = ['t0', 'x0', 'x1', 'c']

        print('Use model:')
        print(self.fit_model._headsummary())

        if obj_ID is None:
            for obj_id, lc in self.sim_lcs.groupby('ID'):
                fit_model.set(z=self.sim_lcs.attrs[obj_id]['zobs'])
                self._set_obj_effects_model(fit_model, obj_id)
                snc_out, snc_mod, params = ut.snc_fitter(self.sim_lcs.loc[obj_id].to_records(),
                                                         fit_model,
                                                         fit_par,
                                                         **kwargs)
                self._fit_res[obj_id] = {'snc_out': snc_out,
                                         'snc_mod': snc_mod,
                                         'params': params}
        else:
            fit_model.set(z=self.sim_lcs.attrs[obj_ID]['zobs'])
            self._set_obj_effects_model(fit_model, obj_ID)

            snc_out, snc_mod, params = ut.snc_fitter(self.sim_lcs.loc[obj_ID].to_records(),
                                                     fit_model,
                                                     fit_par,
                                                     **kwargs)
            self._fit_res[obj_ID] = {'snc_out': snc_out,
                                     'snc_mod': snc_mod,
                                     'params': params}

    def _write_sim(self, write_path, formats=['pkl', 'parquet'], lcs_df=None, sufname=''):
        """Write simulation into a file.

        Parameters
        ----------
        write_path : str
            The output directory.
        formats : lsit(str) or str, opt
            The output formats, 'pkl' or 'fits'.
        lcs_df : pd.dataframe, opt
            A DataFrame containing the lcs to write.
        sufname : str, opt
            A suffix to put behind the file name.

        Returns
        -------
        None
            Write an object.

        """
        if lcs_df is None:
            lcs_df = self.sim_lcs
        header = self.header.copy()
        header['n_obj'] = len(lcs_df.index.levels[0])
        formats = np.atleast_1d(formats)

        io_ut.write_sim(write_path, self.name + sufname, formats, header, lcs_df)

    def write_mod(self, formats=['pkl', 'parquet']):
        """Write a file containing only the modified SN epochs.

        Parameters
        ----------
        formats : list(str) or str, opt
            The output formats, 'pkl' or 'fits'.

        Returns
        -------
        None
            Just write a file.

        """
        self._write_sim(self._dir_path,
                        formats=formats,
                        lcs_df=self.modified_lcs,
                        sufname='_modified')

    def write_fit(self, write_path=None):
        """Write fits results in fits format.

        Returns
        -------
        None
            Write an output file.

        Notes
        -----
        Use write_fit from utils.

        """
        if write_path is None:
            write_path = self._dir_path + self.name + '_fit'

        if self.fit_res is None:
            print('Perform fit before write')
            self.fit_lc()
        if len(self.fit_res) < self.n_obj:
            print('Not all object are fitted')
            for id in self.get('ID'):
                if id not in self.fit_res:
                    self.fit_lc(id)

        meta_keys = ['ID', 'ra', 'dec', 'vpec', 'zpec', 'z2cmb', 'zcos', 'zCMB',
                     'zobs', 'sim_mu', 'com_dist', 'sim_t0', 'm_sct']

        model_name = self.header['model_name']

        if model_name[:4] == 'salt':
            meta_keys += ['sim_x0', 'sim_mb', 'sim_x1', 'sim_c']

        if 'mw_dust' in self.header:
            meta_keys.append('mw_ebv')

        sim_lc_meta = {key: self.get(key) for key in meta_keys}

        if 'sct_mod' in self.header:
            sim_lc_meta['SM_seed'] = self.get(self.header['sct_mod'][:3] + '_RndS')

        io_ut.write_fit(sim_lc_meta, self.fit_res, write_path, sim_meta=self.header)

    def plot_hist(self, key, ax=None, **kwargs):
        """Plot the histogram of the key metadata.

        Parameters
        ----------
        key : str
            A key of lightcurves metadata.
        ax : matplotlib.axis, opt
            ax on which plot the histogram.
        **kwargs : type
            matplotlib plot options.

        Returns
        -------
        type
            Description of returned object.

        """
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            show = True
        ax.hist(self.get(key), **kwargs)
        if show:
            plt.show()

    def plot_lc(self, obj_ID, mag=False, zp=25., plot_sim=True, plot_fit=False, Jy=False,
                mod=False, figsize=(35 / 2.54, 20 / 2.54), dpi=120):
        """Plot the given SN lightcurve.

        Parameters
        ----------
        obj_ID : int
            The Object ID.
        mag : boolean, default = False
            If True plot the magnitude instead of the flux.
        zp : float
            Used zeropoint for the plot.
        plot_sim : boolean, default = True
            If True plot the theorical simulated lightcurve.
        plot_fit : boolean, default = False
            If True plot the fitted lightcurve.
        Jy : boolean, default = False
            If True plot in Jansky.
        mod : boolean, default = False
            If True use the self.modified_lcs rather than self.sim_lcs

        Returns
        -------
        None
            Just plot the SN lightcurve !

        Notes
        -----
        Use plot_lc from utils.

        """
        if mod:
            lc = self.modified_lcs.loc[obj_ID]
            meta = self.modified_lcs.attrs[obj_ID]
        else:
            lc = self.sim_lcs.loc[obj_ID]
            meta = self.sim_lcs.attrs[obj_ID]

        if plot_sim:
            s_model = self.get_obj_sim_model(obj_ID)
        else:
            s_model = None

        if plot_fit and not mod:
            if self.fit_res is None or self.fit_res[obj_ID] is None:
                print('This SN was not fitted, launch fit')
                self.fit_lc(obj_ID)

            if self.fit_res[obj_ID] == 'NaN':
                print('This sn has no fit results')
                return

            f_model = self.fit_res[obj_ID]['snc_mod']
            cov_t0_x0_x1_c = self.fit_res[obj_ID]['snc_out']['covariance'][:, :]
            residuals = True

        elif plot_fit and mod:
            print("You can't fit mod sn, write the in a file and load them"
                  "as SimSample class")
        else:
            f_model = None
            cov_t0_x0_x1_c = None
            residuals = False

        plot_ut.plot_lc(lc,
                        meta,
                        mag=mag,
                        snc_sim_model=s_model,
                        snc_fit_model=f_model,
                        fit_cov=cov_t0_x0_x1_c,
                        zp=zp,
                        residuals=residuals,
                        Jy=Jy,
                        figsize=figsize,
                        dpi=dpi)

    def plot_ra_dec(self, plot_vpec=False, field_dic=None, field_size=None, **kwarg):
        """Plot a mollweide map of ra, dec.

        Parameters
        ----------
        plot_vpec : boolean
            If True plot a vpec colormap.
        field_dic :  dict(int : dict(str : float))
            Dict of fields coordinates -> Field_ID : {'RA', 'Dec'}
        field_size : list(float, float)
            The size of the field [RA, Dec]

        Returns
        -------
        None
            Just plot the map.

        """
        plot_fields = False
        if field_dic is not None and field_size is not None:
            plot_fields = True

        vpec = None

        if plot_vpec:
            vpec = self.get('vpec')

        if plot_fields:
            field_list = self.sim_lcs['fieldID'].unique()

        if plot_fields:
            field_list = np.unique(field_list)
        else:
            field_dic = None
            field_size = None
            field_list = None

        plot_ut.plot_ra_dec(self.get('ra'),
                            self.get('dec'),
                            vpec,
                            field_list,
                            field_dic,
                            field_size,
                            **kwarg)

    def _get_snia_simsncpar(self, ID):
        """Get sncosmo par of one obj for a SNIa model.

        Parameters
        ----------
        ID : int
            Obj ID.

        Returns
        -------
        dict
            Dict containing model parameters values.

        """
        dic_par = {}
        if self._sim_model.source.name[:4] == 'salt':
            dic_par['x0'] = self.meta[ID]['sim_x0']
            dic_par['x1'] = self.meta[ID]['sim_x1']
            dic_par['c'] = self.meta[ID]['sim_c']
        return dic_par

    def get(self, key, mod=False):
        """Get an array of sim_lc metadata.

        Parameters
        ----------
        key : str
            The metadata to access.

        Returns
        -------
        numpy.ndarray
            The array of the key metadata for all SN.

        """
        if mod:
            meta_list = self.modified_lcs.attrs.values()
        else:
            meta_list = self.sim_lcs.attrs.values()
        return np.array([meta[key] for meta in meta_list])

    @property
    def name(self):
        """Get sample name."""
        return self._name

    @property
    def header(self):
        """Get header."""
        return self._header

    @property
    def n_obj(self):
        """Get SN number."""
        return len(self.sim_lcs.index.levels[0])

    @property
    def sim_lcs(self):
        """Get sim_lcs."""
        return self._sim_lcs

    @property
    def meta(self):
        """Get lcs meta dict."""
        return self._sim_lcs.attrs

    @property
    def sim_model(self):
        """Get sim model."""
        return self._sim_model

    @property
    def fit_model(self):
        """Get fit model."""
        return self._fit_model

    @property
    def fit_res(self):
        """Get fit results list."""
        return self._fit_res
