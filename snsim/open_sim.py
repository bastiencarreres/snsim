"""OpenSim class used to open simulations file."""

import os
import pickle
import numpy as np
from astropy.io import fits
from astropy.table import Table
from . import utils as ut
from . import scatter as sct
from . import plot_utils as plot_ut
from . import dust_utils as dst_ut


class OpenSim:
    """This class allow to open simulation file, make plot and run the fit.

    Parameters
    ----------
    sim_file : str
        Path to the simulation file fits/pkl.
    model_dir : str
        Path to the .model used during simulation

    Attributes
    ----------
    _file_path : str
        The path of the simulation file.
    _file_ext : str
        sim file extension.
    _sim_lc : list(astropy.Table)
        List containing the simulated lightcurves.
    _header : dict
        A dict containing simulation meta.
    _model_dir : str
        A copy of input model dir.
    _fit_model : sncosmo.Model
        The model used to fit the lightcurves.
    _fit_res : list(sncomso.utils.Result)
        The reuslts of sncosmo fit.

    Methods
    -------
    _init_sim_lc()
        Extract data from file.
    plot_lc(sn_ID, mag = False, zp = 25., plot_sim = True, plot_fit = False)
        Plot the given SN lightcurve.
    plot_ra_dec(self, plot_vpec=False, **kwarg):
        Plot a mollweide map of ra, dec.
    fit_lc(sn_ID = None)
        Fit all or just one SN lightcurve(s).
    write_fit()
        Write fits results in fits format.


    """

    def __init__(self, sim_file, model_dir):
        """Copy some function of snsim to allow to use sim file."""
        self._file_path, self._file_ext = os.path.splitext(sim_file)
        self._sn = None
        self._sim_lc = None
        self._header = None
        self._init_sim_lc()
        self._model_dir = model_dir
        self._fit_model = ut.init_sn_model(self.header['Mname'], model_dir)
        self._fit_res = None
        self._fit_resmod = None

    def _init_sim_lc(self):
        if self._file_ext == '.fits':
            sim_lc = []
            with fits.open(self._file_path + self._file_ext) as sf:
                header = sf[0].header
                for hdu in sf[1:]:
                    data = hdu.data
                    tab = Table(data)
                    tab.meta = hdu.header
                    sim_lc.append(tab)
            self._sim_lc = sim_lc
            self._header = header

        elif self._file_ext == '.pkl':
            with open(self._file_path + self._file_ext, 'rb') as f:
                self._sn = pickle.load(f)

    @property
    def sn(self):
        """Get SnSimPkl object."""
        if self._sn is None:
            print('You open a fits file => No SnSimPkl object')
            return None
        else:
            return self._sn

    @property
    def sim_lc(self):
        """Get sim_lc list."""
        if self._sim_lc is None:
            return self.sn.sim_lc
        return self._sim_lc

    @property
    def header(self):
        """Get header dict."""
        if self._header is None:
            return self.sn.header
        return self._header

    @property
    def fit_res(self):
        """Get fit results list."""
        return self._fit_res

    @property
    def fit_resmod(self):
        """Get fit sncosmo model results."""
        return self._fit_resmod

    def fit_lc(self, sn_ID=None, mw_dust=-2):
        """Fit all or just one SN lightcurve(s).

        Parameters
        ----------
        sn_ID : int, default is None
            The SN ID, if not specified all SN are fit.

        Returns
        -------
        None
            Directly modified the _fit_res attribute.

        Notes
        -----
        Use snc_fitter from utils

        """
        if self._fit_res is None:
            self._fit_res = [None] * len(self.sim_lc)
            self._fit_resmod = [None] * len(self.sim_lc)
        fit_model = self._fit_model.__copy__()
        model_name = self.header['Mname']
        if model_name in ('salt2', 'salt3'):
            fit_par = ['t0', 'x0', 'x1', 'c']

        mw_mod = None
        if mw_dust == -2 and 'mwd_mod' in self.header:
            mw_mod = [self.header['mwd_mod'], self.header['mw_rv']]
        elif isinstance(mw_dust, (str, list, np.ndarray)):
            mw_mod = mw_dust
        else:
            print('Do not use mw dust')

        if mw_mod is not None:
            dst_ut.init_mw_dust(fit_model, mw_mod)
            if isinstance(mw_mod, (list, np.ndarray)):
                rv = mw_mod[1]
                mod_name = mw_mod[0]
            else:
                rv = 3.1
                mod_name = mw_mod
            print(f'Use MW dust model {mod_name} with RV = {rv}')

        if sn_ID is None:
            for i, lc in enumerate(self.sim_lc):
                if self._fit_res[i] is None:
                    fit_model.set(z=lc.meta['z'])
                    if mw_dust is not None:
                        dst_ut.add_mw_to_fit(fit_model, lc.meta['mw_ebv'], mod_name, rv=rv)
                    self._fit_res[i], self._fit_resmod[i] = ut.snc_fitter(lc, fit_model, fit_par)
        else:
            fit_model.set(z=self.sim_lc[sn_ID].meta['z'])
            if mw_dust is not None:
                dst_ut.add_mw_to_fit(fit_model, self.sim_lc[sn_ID].meta['mw_ebv'], mod_name, rv=rv)
            self._fit_res[sn_ID], self._fit_resmod[sn_ID] = ut.snc_fitter(self.sim_lc[sn_ID],
                                                                          fit_model,
                                                                          fit_par)

    def plot_lc(self, sn_ID, mag=False, zp=25., plot_sim=True, plot_fit=False, Jy=False):
        """Plot the given SN lightcurve.

        Parameters
        ----------
        sn_ID : int
            The Supernovae ID.
        mag : boolean, default = False
            If True plot the magnitude instead of the flux.
        zp : float
            Used zeropoint for the plot.
        plot_sim : boolean, default = True
            If True plot the theorical simulated lightcurve.
        plot_fit : boolean, default = False
            If True plot the fitted lightcurve.

        Returns
        -------
        None
            Just plot the SN lightcurve !

        Notes
        -----
        Use plot_lc from utils.

        """
        lc = self.sim_lc[sn_ID]

        if plot_sim:
            model_name = self.header['Mname']

            s_model = ut.init_sn_model(model_name, self._model_dir)

            dic_par = {'z': lc.meta['z'],
                       't0': lc.meta['sim_t0']}

            if model_name in ('salt2', 'salt3'):
                dic_par['x0'] = lc.meta['sim_x0']
                dic_par['x1'] = lc.meta['sim_x1']
                dic_par['c'] = lc.meta['sim_c']
            s_model.set(**dic_par)

            if 'Smod' in self.header:
                sct.init_sn_sct_model(s_model, self.header['Smod'])
                par_rd_name = self.header['Smod'][:3] + '_RndS'
                s_model.set(**{par_rd_name: lc.meta[par_rd_name]})

            if 'mwd_mod' in self.header:
                dst_ut.init_mw_dust(s_model, [self.header['mwd_mod'], self.header['mw_rv']])
                if self.header['mwd_mod'].lower() not in ['f99']:
                    s_model.set(mw_r_v=lc.meta['mw_r_v'])
                s_model.set(mw_ebv=lc.meta['mw_ebv'])
        else:
            s_model = None

        if plot_fit:
            if self.fit_res is None or self.fit_res[sn_ID] is None:
                print('This SN was not fitted, launch fit')
                if 'mwd_mod' in self.header:
                    print('Use sim input mw dust')
                    if 'mw_rv' in self.header:
                        mw_dust = [self.header['mwd_mod'], self.header['mw_rv']]
                    else:
                        mw_dust = [self.header['mwd_mod'], 3.1]
                self.fit_lc(sn_ID, mw_dust=mw_dust)
            elif self.fit_res[sn_ID] is None:
                print('This SN was not fitted, launch fit')
                self.fit_lc(sn_ID)

            if self.fit_res[sn_ID] == 'NaN':
                print('This sn has no fit results')
                return

            f_model = self.fit_resmod[sn_ID]
            cov_t0_x0_x1_c = self.fit_res[sn_ID]['covariance'][:, :]
            residuals = True
        else:
            f_model = None
            cov_t0_x0_x1_c = None
            residuals = False

        plot_ut.plot_lc(self.sim_lc[sn_ID],
                        mag=mag,
                        snc_sim_model=s_model,
                        snc_fit_model=f_model,
                        fit_cov=cov_t0_x0_x1_c,
                        zp=zp,
                        residuals=residuals,
                        Jy=Jy)

    def plot_ra_dec(self, plot_vpec=False, **kwarg):
        """Plot a mollweide map of ra, dec.

        Parameters
        ----------
        plot_vpec : boolean
            If True plot a vpec colormap.

        Returns
        -------
        None
            Just plot the map.

        """
        ra = []
        dec = []
        vpec = None
        if plot_vpec:
            vpec = []
        for lc in self.sim_lc:
            ra.append(lc.meta['ra'])
            dec.append(lc.meta['dec'])
            if plot_vpec:
                vpec.append(lc.meta['vpec'])
        plot_ut.plot_ra_dec(np.asarray(ra),
                            np.asarray(dec),
                            vpec,
                            **kwarg)

    def write_fit(self):
        """Write fits results in fits format.

        Returns
        -------
        None
            Write an output file.

        Notes
        -----
        Use write_fit from utils.

        """
        if self.fit_res is None:
            print('Perform fit before write')
            self.fit_lc()
        for i, res in enumerate(self.fit_res):
            if res is None:
                self.fit_lc(self.sim_lc[i].meta['sn_id'])

        sim_lc_meta = {'sn_id': [lc.meta['sn_id'] for lc in self.sim_lc],
                       'ra': [lc.meta['ra'] for lc in self.sim_lc],
                       'dec': [lc.meta['dec'] for lc in self.sim_lc],
                       'vpec': [lc.meta['vpec'] for lc in self.sim_lc],
                       'zpec': [lc.meta['zpec'] for lc in self.sim_lc],
                       'z2cmb': [lc.meta['z2cmb'] for lc in self.sim_lc],
                       'zcos': [lc.meta['zcos'] for lc in self.sim_lc],
                       'zCMB': [lc.meta['zCMB'] for lc in self.sim_lc],
                       'zobs': [lc.meta['z'] for lc in self.sim_lc],
                       'sim_mu': [lc.meta['sim_mu'] for lc in self.sim_lc]}

        model_name = self.header['Mname']
        if model_name in ('salt2', 'salt3'):
            sim_lc_meta['sim_mb'] = [lc.meta['sim_mb'] for lc in self.sim_lc]
            sim_lc_meta['sim_x1'] = [lc.meta['sim_x1'] for lc in self.sim_lc]
            sim_lc_meta['sim_c'] = [lc.meta['sim_c'] for lc in self.sim_lc]
            sim_lc_meta['m_sct'] = [lc.meta['m_sct'] for lc in self.sim_lc]

        if 'Smod' in self.header:
            sim_lc_meta['SM_seed'] = [lc.meta[self.header['Smod'][:3] + '_RndS']
                                      for lc in self.sim_lc]

        write_file = self._file_path + '_fit.fits'
        ut.write_fit(sim_lc_meta, self.fit_res, write_file, sim_meta=self.header)
