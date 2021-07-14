"""This module contains usefull function for the simulation"""

import numpy as np
import sncosmo as snc
from astropy.table import Table
import astropy.time as atime
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from . import nb_fun as nbf
from . import salt_utils as salt_ut
from .constants import C_LIGHT_KMS

def init_astropy_time(date):
    """Take a date and give a astropy.time.Time object.

    Parameters
    ----------
    date : int, float or str
        The date in MJD number or YYYY-MM-DD string.

    Returns
    -------
    astropy.time.Time
        An astropy.time Time object of the given date.

    """
    if isinstance(date, (int, float)):
        date_format = 'mjd'
    elif isinstance(date, str):
        date_format = 'iso'
    return atime.Time(date, format=date_format)

def compute_z_cdf(z_shell, shell_time_rate):
    """Compute the cumulative distribution function of redshift.

    Parameters
    ----------
    z_shell : numpy.ndarray(float)
        The redshift of the shell edges.
    shell_time_rate : numpy.ndarray(float)
        The time rate of each shell.

    Returns
    -------
    list(numpy.ndarray(float), numpy.ndarray(float))
        redshift, CDF(redshift).

    """
    dist = np.append(0, np.cumsum(shell_time_rate))
    norm = dist[-1]
    return [z_shell, dist/norm]

def is_asym(sigma):
    """Check if sigma represents an asymetric distribution.

    Parameters
    ----------
    sigma : flaot or list
        The sigma parameter(s) of the Gaussian.

    Returns
    -------
    tuple
        sigma low and sigma high of an asymetric Gaussian.

    """
    sigma = np.atleast_1d(sigma)
    if sigma.size == 2:
        return sigma
    return sigma[0], sigma[0]

def asym_gauss(mean, sig_low, sig_high=None, rand_gen=None):
    """Generate random parameters using an asymetric Gaussian distribution.

    Parameters
    ----------
    mean : float
        The central value of the Gaussian.
    sig_low : float
        The low sigma.
    sig_high : float
        The high sigma.
    rand_gen : numpy.random.default_rng, optional
        Numpy random generator.

    Returns
    -------
    float
        Random variable.

    """

    if sig_high is None:
        sig_high = sig_low
    if rand_gen is None:
        low_or_high = np.random.random()
        nbr = abs(np.random.normal())
    else:
        low_or_high = rand_gen.random()
        nbr = abs(rand_gen.normal())
    if low_or_high < sig_low/(sig_high+sig_low):
        nbr *= -sig_low
    else:
        nbr *= sig_high
    return mean + nbr


def is_same_cosmo_model(dic, astropy_model):
    """Check if cosmo parameters in a dic are the same used in astropy_model.

    Parameters
    ----------
    dic : dict
        Contain som cosmological parameters.
    astropy_model : astropy.cosmology
        An astropy cosmological model.

    Returns
    -------
    type
        Description of returned object.

    """
    for k, v in dic.items():
        if v != astropy_model.__dict__['_'+k]:
            return False
    return True


def compute_z2cmb(ra, dec, cmb):
    """Compute the redshifts of a list of objects relative to the CMB.

    Parameters
    ----------
    ra : np.ndarray(float)
        Right Ascension of the objects.
    dec : np.ndarray(float)
        Declinaison of the objects.
    cmb : dict
        Dict containing cmb coords and velocity.

    Returns
    -------
    np.ndarray(float)
        Redshifts relative to cmb.

    """
    l_cmb = cmb['l_cmb']
    b_cmb = cmb['b_cmb']
    v_cmb = cmb['v_cmb']

    # use ra dec to simulate the effect of our motion
    coordfk5 = SkyCoord(ra * u.rad,
                        dec * u.rad,
                        frame='fk5')  # coord in fk5 frame

    galac_coord = coordfk5.transform_to('galactic')
    l_gal = galac_coord.l.rad - 2 * np.pi * \
        np.sign(galac_coord.l.rad) * (abs(galac_coord.l.rad) > np.pi)
    b_gal = galac_coord.b.rad

    ss = np.sin(b_gal) * np.sin(b_cmb * np.pi / 180)
    ccc = np.cos(b_gal) * np.cos(b_cmb * np.pi / 180) * np.cos(l_gal - l_cmb * np.pi / 180)
    return (1 - v_cmb * (ss + ccc) / C_LIGHT_KMS) - 1.


def init_sn_model(name, model_dir):
    """Initialise a sncosmo model.

    Parameters
    ----------
    name : str
        Name of the model.
    model_dir : str
        PAth to the model files.

    Returns
    -------
    sncosmo.Model
        sncosmo Model corresponding to input configuration.
    """
    if name == 'salt2':
        return snc.Model(source=snc.SALT2Source(model_dir, name='salt2'))
    elif name == 'salt3':
        return snc.Model(source=snc.SALT3Source(model_dir, name='salt3'))
    return None


def snc_fitter(lc, fit_model, fit_par):
    """Fit a given lightcurve with sncosmo.

    Parameters
    ----------
    lc : astropy.Table
        The SN lightcurve.
    fit_model : sncosmo.Model
        Model used to fit the ligthcurve.
    fit_par : list(str)
        The parameters to fit.

    Returns
    -------
    sncosmo.utils.Result (numpy.nan if no result)
        sncosmo dict of fit results.

    """
    try:
        res = snc.fit_lc(lc, fit_model, fit_par, modelcov=True)
    except BaseException:
        res = ['NaN', 'NaN']
    return res


def norm_flux(flux_table, zp):
    """Rescale the flux to a given zeropoint.

    Parameters
    ----------
    flux_table : astropy.Table
        A table containing at least flux and fluxerr.
    zp : float
        The zeropoint to rescale the flux.

    Returns
    -------
    np.ndarray(float), np.ndarray(float)
        Rescaled flux and fluxerr arry.

    """
    norm_factor = 10**(0.4 * (zp - flux_table['zp']))
    flux_norm = flux_table['flux'] * norm_factor
    fluxerr_norm = flux_table['fluxerr'] * norm_factor
    return flux_norm, fluxerr_norm

def flux_to_Jansky(zp, band):
    """Give the factor to convert flux in uJy.

    Parameters
    ----------
    zp : float
        The actual zeropoint of flux.
    band : str
        The sncosmo band in which compute the factor.

    Returns
    -------
    float
        The conversion factor.

    """
    magsys = snc.get_magsystem('ab')
    b = snc.get_bandpass(band)
    nu, dnu = snc.utils.integration_grid(snc.constants.C_AA_PER_S/b.maxwave(),
                                         snc.constants.C_AA_PER_S/b.minwave(),
                                         snc.constants.C_AA_PER_S/snc.constants.MODEL_BANDFLUX_SPACING)

    trans = b(snc.constants.C_AA_PER_S/nu)
    trans_int = np.sum(trans / nu) * dnu / snc.constants.H_ERG_S
    norm = 10**(-0.4*zp) * magsys.zpbandflux(b) / trans_int * 10**23 * 10**6
    return norm

def change_sph_frame(ra, dec, ra_frame, dec_frame):
    """Compute object coord in a new frame.

    Parameters
    ----------
    ra : float
        Right Ascension of the object.
    dec : float
        Declinaison of the oject.
    ra_frame : float/numpy.ndarray(float)
        Right Ascension of frame in which compute the new coords.
    dec_frame : float/numpy.ndarray(float)
        Declinaison of frame in which compute the new coords.

    Returns
    -------
    numpy.ndaray(float), numpy.ndaray(float)
        Object coordinates in each of the frames.

    """
    if isinstance(ra_frame, float):
        ra_frame = np.array([ra_frame])
        dec_frame = np.array([dec_frame])
    vec = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
    new_ra, new_dec = nbf.new_coord_on_fields(ra_frame, dec_frame, vec)
    return new_ra, new_dec


def write_fit(sim_lc_meta, fit_res, directory, sim_meta={}):
    """Write fit into a fits file.

    Parameters
    ----------
    sim_lc_meta : dict{list}
        Meta data of all lightcurves.
    fit_res : list(sncosmo.utils.Result)
        List of sncosmo fit results for each lightcurve.
    directory : str
        Destination of write file.
    sim_meta : dict
        General simulation meta data.

    Returns
    -------
    None
        Just write a file.

    """
    data = sim_lc_meta.copy()

    fit_keys = ['t0', 'e_t0',
                'chi2', 'ndof']
    MName = sim_meta['Mname']

    if MName in ('salt2', 'salt3'):
        fit_keys += ['x0', 'e_x0', 'mb', 'e_mb', 'x1',
                     'e_x1', 'c', 'e_c', 'cov_x0_x1', 'cov_x0_c',
                     'cov_mb_x1', 'cov_mb_c', 'cov_x1_c']

    for k in fit_keys:
        data[k] = []

    for res in fit_res:
        if res != 'NaN':
            par = res['parameters']
            data['t0'].append(par[1])
            data['e_t0'].append(np.sqrt(res['covariance'][0, 0]))

            if MName in ('salt2', 'salt3'):
                par_cov = res['covariance'][1:, 1:]
                mb_cov = salt_ut.cov_x0_to_mb(par[2], par_cov)
                data['x0'].append(par[2])
                data['e_x0'].append(np.sqrt(par_cov[0, 0]))
                data['mb'].append(salt_ut.x0_to_mB(par[2]))
                data['e_mb'].append(np.sqrt(mb_cov[0, 0]))
                data['x1'].append(par[3])
                data['e_x1'].append(np.sqrt(par_cov[1, 1]))
                data['c'].append(par[4])
                data['e_c'].append(np.sqrt(par_cov[2, 2]))
                data['cov_x0_x1'].append(par_cov[0, 1])
                data['cov_x0_c'].append(par_cov[0, 2])
                data['cov_x1_c'].append(par_cov[1, 2])
                data['cov_mb_x1'].append(mb_cov[0, 1])
                data['cov_mb_c'].append(mb_cov[0, 2])

            data['chi2'].append(res['chisq'])
            data['ndof'].append(res['ndof'])
        else:
            for k in fit_keys:
                data[k].append(np.nan)

    for k, v in sim_lc_meta.items():
        data[k] = v

    table = Table(data)

    hdu = fits.table_to_hdu(table)
    hdu_list = fits.HDUList([fits.PrimaryHDU(header=fits.Header(sim_meta)), hdu])
    hdu_list.writeto(directory, overwrite=True)
    print(f'Fit result output file : {directory}')
