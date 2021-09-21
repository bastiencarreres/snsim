"""SNSimSample class used to store simulations."""

import os
import pickle
import numpy as np
from astropy.io import fits
from astropy.table import Table
from . import utils as ut
from . import scatter as sct
from . import plot_utils as plot_ut
from . import dust_utils as dst_ut


class SNSimSample:
    """Class to store simulation as pickle.

    Parameters
    ----------
    sample_name : str
        Name of the sample.
    sim_lcs : list(astropy.Table)
        Lightcurves of the SNs.
    header : OrderedDic()
        The simulation configuration.
    model_dir : str, opt
        The directory of the configuration files of the sim model.
    file_path : str, opt
        The path to the simualation directory.

    Returns
    -------
    type
        Description of returned object.
    """

    def __init__(self, sample_name, sim_lcs, header, model_dir=None, file_path=None):

        """Initialize SNSimSample class."""
        self._name = sample_name
        self._header = header
        self._sim_lcs = sim_lcs
        self._model_dir = model_dir
        self._file_path = file_path

        self._fit_model = ut.init_sn_model(self.header['Mname'], model_dir)
        self._fit_res = None
        self._fit_resmod = None

        self._select_lcs = None

        self._bands = []
        for lc in sim_lcs:
            for b in lc['band']:
                if b not in self._bands:
                    self._bands.append(b)

    @classmethod
    def fromFile(cls, sim_file, model_dir=None):
        """Initialize the class from a fits or pickle file.

        Parameters
        ----------
        cls : SNSimSample class
            The SNSimSample class.
        sim_file : str
            The file to load.
        model_dir : str, opt
            The directory of the configuration files of the sim model.

        Returns
        -------
        SNSimSample class object
            A SNSimSample class with the simulated lcs.

        """
        file_path, file_ext = os.path.splitext(sim_file)
        sample_name = os.path.basename(file_path)
        if file_ext == '.fits':
            sim_lcs = []
            with fits.open(file_path + file_ext) as sf:
                header = sf[0].header
                for hdu in sf[1:]:
                    data = hdu.data
                    tab = Table(data)
                    tab.meta = hdu.header
                    sim_lcs.append(tab)
            return cls(sample_name, sim_lcs, header, model_dir=model_dir, file_path=file_path)

        elif file_ext == '.pkl':
            with open(file_path + file_ext, 'rb') as f:
                return pickle.load(f)

    @property
    def name(self):
        """Get sample name."""
        return self._name

    @property
    def header(self):
        """Get header."""
        return self._header

    @property
    def n_sn(self):
        """Get SN number."""
        return len(self._sim_lcs)

    @property
    def sim_lcs(self):
        """Get sim_lc."""
        return self._sim_lcs

    @property
    def fit_res(self):
        """Get fit results list."""
        return self._fit_res

    @property
    def fit_resmod(self):
        """Get fit sncosmo model results."""
        return self._fit_resmod

    @property
    def select_lcs(self):
        """Get selected lcs."""
        return self._select_lcs

    def SNR_select(self,
                   selec_function,
                   SNR_mean=5,
                   SNR_limit=[15, 0.99],
                   randseed=np.random.randint(1000, 100000)):
        r"""Run a SNR efficiency detection on all lcs.

        Parameters
        ----------
        selec_function : str
            Can be 'approx' function TODO : add interpolation for function from file.
        SNR_mean : float or dic
            The SNR for which the detection probability is 1/2 -> SNR_mean.
        SNR_limit : list of dic(list)
            A SNR and its probability of detection -> $SNR_p$ and p.
        randseed : int
            Randseed for random detection.

        Returns
        -------
        None
            Just fill the self._select_lcs attribute.

        Notes
        -----
        The detection probability function :
        .. math::
            P_{det}(SNR) = \frac{1}{1+\left(\frac{SNR_{mean}}{SNR}\right)^n}

        where :math:`n = \frac{\ln\left(\frac{1-p}{p}\right)}{\ln(SNR_{mean}) - \ln(SNR_p)}`

        """
        rand_gen = np.random.default_rng(randseed)
        self._select_lcs = []
        SNR_proba = {}
        if selec_function == 'approx':
            if isinstance(SNR_limit, (list, np.ndarray)) and isinstance(SNR_mean, (int, float)):
                for b in self._bands:
                    SNR_proba[b] = lambda SNR: ut.SNR_pdet(SNR,
                                                           SNR_mean,
                                                           SNR_limit[0],
                                                           SNR_limit[1])
            else:
                for b in self._bands:
                    SNR_proba[b] = lambda SNR: ut.SNR_pdet(SNR,
                                                           SNR_mean[b],
                                                           SNR_limit[b][0],
                                                           SNR_limit[b][1])

        for i, lc in enumerate(self.sim_lcs):
            selec_mask = np.zeros(len(lc), dtype='bool')
            SNR = lc['flux']/lc['fluxerr']
            for b in self._bands:
                bmask = lc['band'] == b
                selec_mask[bmask] = rand_gen.random(np.sum(bmask)) <= SNR_proba[b](SNR[bmask])
            if np.sum(selec_mask) > 0:
                self._select_lcs.append(self.sim_lcs[i][selec_mask])

    def get(self, key):
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
        return np.array([lc.meta[key] for lc in self.sim_lcs])

    def _write_sim(self, write_path, formats=['pkl', 'fits'], lcs_list=None, sufname=''):
        """Write simulation into a file.

        Parameters
        ----------
        write_path : str
            The output directory.
        formats : lsit(str) or str, opt
            The output formats, 'pkl' or 'fits'.
        lcs_list : np.ndarray(astropy.Table), opt
            A table containing the lcs to write.
        sufname : str, opt
            A suffix to put behind the file name.

        Returns
        -------
        None
            Write an object.

        """
        if lcs_list is None:
            lcs_list = self.sim_lcs

        formats = np.atleast_1d(formats)
        if 'fits' in formats:
            lc_hdu_list = (fits.table_to_hdu(lc) for lc in lcs_list)
            hdu_list = fits.HDUList(
                [fits.PrimaryHDU(header=fits.Header(self.header))] + list(lc_hdu_list))

            hdu_list.writeto(write_path + self.name + sufname + '.fits',
                             overwrite=True)

        # Export lcs as pickle
        if 'pkl' in formats:
            with open(write_path + self.name + sufname + '.pkl', 'wb') as file:
                pickle.dump(SNSimSample(self._name + sufname,
                                        lcs_list,
                                        self._header,
                                        self._model_dir,
                                        self._file_path),
                            file)

    def write_select(self, formats=['pkl', 'fits']):
        """Write a file containing only the selected SN epochs.

        Parameters
        ----------
        formats : list(str) or str, opt
            The output formats, 'pkl' or 'fits'.

        Returns
        -------
        None
            Just write a file.

        """
        self._write_sim(self._file_path,
                        formats=formats,
                        lcs_list=self.select_lcs,
                        sufname='_selected')

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
            self._fit_res = [None] * len(self.sim_lcs)
            self._fit_resmod = [None] * len(self.sim_lcs)
            self._fit_dic = [None] * len(self.sim_lcs)

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
            for i, lc in enumerate(self.sim_lcs):
                if self._fit_res[i] is None:
                    fit_model.set(z=lc.meta['zobs'])
                    if mw_dust is not None:
                        dst_ut.add_mw_to_fit(fit_model, lc.meta['mw_ebv'], mod_name, rv=rv)
                    self._fit_res[i], self._fit_resmod[i], self._fit_dic[i] = ut.snc_fitter(
                                                                                     lc,
                                                                                     fit_model,
                                                                                     fit_par)
        else:
            fit_model.set(z=self.sim_lcs[sn_ID].meta['zobs'])
            if mw_dust is not None:
                dst_ut.add_mw_to_fit(fit_model, self.sim_lcs[sn_ID].meta['mw_ebv'], mod_name, rv=rv)
            self._fit_res[sn_ID], self._fit_resmod[sn_ID], self._fit_dic[sn_ID] = ut.snc_fitter(
                                                                                self.sim_lcs[sn_ID],
                                                                                fit_model,
                                                                                fit_par)

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
            write_path = self._file_path + '_fit.fits'

        if self.fit_res is None:
            print('Perform fit before write')
            self.fit_lc()
        for i, res in enumerate(self.fit_res):
            if res is None:
                self.fit_lc(self.sim_lcs[i].meta['sn_id'])

        sim_lc_meta = {'sn_id': self.get('sn_id'),
                       'ra': self.get('ra'),
                       'dec': self.get('dec'),
                       'vpec': self.get('vpec'),
                       'zpec': self.get('zpec'),
                       'z2cmb': self.get('z2cmb'),
                       'zcos': self.get('zcos'),
                       'zCMB': self.get('zCMB'),
                       'zobs': self.get('zobs'),
                       'sim_mu': self.get('sim_mu'),
                       'com_dist': self.get('com_dist'),
                       'sim_t0': self.get('sim_t0'),
                       'm_sct': self.get('m_sct')}

        model_name = self.header['Mname']

        if model_name in ('salt2', 'salt3'):
            sim_lc_meta['sim_x0'] = self.get('sim_x0')
            sim_lc_meta['sim_mb'] = self.get('sim_mb')
            sim_lc_meta['sim_x1'] = self.get('sim_x1')
            sim_lc_meta['sim_c'] = self.get('sim_c')

        if 'Smod' in self.header:
            sim_lc_meta['SM_seed'] = self.get(self.header['Smod'][:3] + '_RndS')

        if 'mw_dust' in self.header:
            sim_lc_meta['MW_EBV'] = self.get('mw_ebv')

        ut.write_fit(sim_lc_meta, self.fit_res, self._fit_dic, write_path, sim_meta=self.header)

    def plot_lc(self, sn_ID, mag=False, zp=25., plot_sim=True, plot_fit=False, Jy=False,
                selected=False):
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
        Jy : boolean, default = False
            If True plot in Jansky.
        selected : boolean, default = False
            If True use the self.select_lcs rather than self.sim_lcs

        Returns
        -------
        None
            Just plot the SN lightcurve !

        Notes
        -----
        Use plot_lc from utils.

        """
        if selected:
            lc = self.select_lcs[sn_ID]
        else:
            lc = self.sim_lcs[sn_ID]

        if plot_sim:
            model_name = self.header['Mname']

            s_model = ut.init_sn_model(model_name, self._model_dir)

            dic_par = {'z': lc.meta['zobs'],
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
                else:
                    mw_dust = None
                self.fit_lc(sn_ID, mw_dust=mw_dust)

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

        plot_ut.plot_lc(self.sim_lcs[sn_ID],
                        mag=mag,
                        snc_sim_model=s_model,
                        snc_fit_model=f_model,
                        fit_cov=cov_t0_x0_x1_c,
                        zp=zp,
                        residuals=residuals,
                        Jy=Jy)

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
        if field_dic is not None and field_size is not None:
            plot_fields = True

        ra = []
        dec = []
        vpec = None
        if plot_vpec:
            vpec = []

        if plot_fields:
            field_list = []

        for lc in self.sim_lcs:
            ra.append(lc.meta['ra'])
            dec.append(lc.meta['dec'])
            if plot_vpec:
                vpec.append(lc.meta['vpec'])
            if plot_fields:
                field_list = np.concatenate((field_list, np.unique(lc['fieldID'])))

        if plot_fields:
            field_list = np.unique(field_list)
        else:
            field_dic = None
            field_size = None
            field_list = None

        plot_ut.plot_ra_dec(np.asarray(ra),
                            np.asarray(dec),
                            vpec,
                            field_list,
                            field_dic,
                            field_size,
                            **kwarg)
