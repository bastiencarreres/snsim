"""This module contains usefull function for the simulation"""

import sncosmo as snc
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from numpy import power as pw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from . import nb_fun as nbf
from .constants import SNC_MAG_OFFSET_AB, C_LIGHT_KMS


def x0_to_mB(x0):
    """Convert SALT x0 to bessellB restframe magnitude.

    Parameters
    ----------
    x0 : float
        SALT normalisation factor.

    Returns
    -------
    float or numpy.ndarray(float)
        Array or value of corresponding bessell B magnitude.

    """
    return -2.5 * np.log10(x0) + SNC_MAG_OFFSET_AB


def mB_to_x0(mB):
    """Convert restframe bessellB magnitude into SALT x0.

    Parameters
    ----------
    mB : float
        Restframe bessellB magnitude.

    Returns
    -------
    float
        SALT x0 parameter.

    """
    return pw(10, -0.4 * (mB - SNC_MAG_OFFSET_AB))


def cov_x0_to_mb(x0, cov):
    """Convert x0,x1,c covariance into mB,x1,c covariance.

    Parameters
    ----------
    x0 : float
        SALT x0 parameter.
    cov : numpy.array(float, size = (3,3))
        SALT x0,x1,c covariance matrix

    Returns
    -------
    numpy.array(float, size = (3,3))
        SALT mb,x1,c covariance matrix.

    """
    J = np.array([[-2.5 / np.log(10) * 1 / x0, 0, 0], [0, 1, 0], [0, 0, 1]])
    new_cov = J @ cov @ J.T
    return new_cov


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
    ra_cmb = cmb['ra_cmb']
    dec_cmb = cmb['dec_cmb']
    v_cmb = cmb['v_cmb']

    # use ra dec to simulate the effect of our motion
    coordfk5 = SkyCoord(ra * u.rad,
                        dec * u.rad,
                        frame='fk5')  # coord in fk5 frame

    galac_coord = coordfk5.transform_to('galactic')
    ra_gal = galac_coord.l.rad - 2 * np.pi * \
        np.sign(galac_coord.l.rad) * (abs(galac_coord.l.rad) > np.pi)
    dec_gal = galac_coord.b.rad

    ss = np.sin(dec_gal) * np.sin(dec_cmb * np.pi / 180)
    ccc = np.cos(dec_gal) * np.cos(dec_cmb * np.pi /
                                   180) * np.cos(ra_gal - ra_cmb * np.pi / 180)
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
        res = snc.fit_lc(lc, fit_model, fit_par, modelcov=True)[0]
    except BaseException:
        res = np.nan
    return res


def compute_salt_fit_error(fit_model, cov, band, time_th, zp, magsys='ab'):
    """Compute fit error on flux from sncosmo fit covariance x0,x1,c.

    Parameters
    ----------
    fit_model : sncosmo.Model
        The model used to fit the sn lightcurve.
    cov : numpy.ndarray(float, size=(3,3))
        sncosmo x0,x1,c covariance matrix from SALT fit.
    band : str
        The band in which the error is computed.
    time_th : numpy.ndaray(float)
        Time for which compute the flux error.
    zp : float
        zeropoint to scale the error.
    magsys : str
        Magnitude system to use.

    Returns
    -------
    numpy.ndarray(float)
        Flux error for each input time.

    Notes
    -----
    Compute theorical fluxerr from fit err = sqrt(COV)
    where COV = J**T * COV(x0,x1,c) * J with J = (dF/dx0, dF/dx1, dF/dc) the jacobian.
    According to Fnorm = x0/(1+z) * int_\\lambda (M0(lambda_s,p)+x1*M1(lambda_s,p))*10**(-0.4*c*CL(lambda_s)) * T_b(lambda) * lambda/hc dlambda * norm_factor
    where norm_factor = 10**(0.4*ZP_norm)/ZP_magsys. We found :
    dF/dx0 = F/x0
    dF/dx1 = x0/(1+z) * int_lambda M1(lambda_s,p))*10**(-0.4*c*CL(lambda_s)) * T_b(lambda) * lambda/hc dlambda * norm_factor
    dF/dc  =  -0.4*ln(10)*x0/(1+z) * int_\\lambda (M0(lambda_s,p)+x1*M1(lambda_s,p))*CL(lambda_s)*10**(-0.4*c*CL(lambda_s)) * T_b(lambda) * lambda/hc dlambda * norm_factor

    """
    a = 1. / (1 + fit_model.parameters[0])
    t0 = fit_model.parameters[1]
    x0 = fit_model.parameters[2]
    x1 = fit_model.parameters[3]
    c = fit_model.parameters[4]
    b = snc.get_bandpass(band)
    wave, dwave = snc.utils.integration_grid(
        b.minwave(), b.maxwave(), snc.constants.MODEL_BANDFLUX_SPACING)
    trans = b(wave)
    ms = snc.get_magsystem(magsys)
    zpms = ms.zpbandflux(b)
    normfactor = 10**(0.4 * zp) / zpms
    M1 = fit_model.source._model['M1']
    M0 = fit_model.source._model['M0']
    CL = fit_model.source._colorlaw

    p = time_th - t0

    dfdx0 = fit_model.bandflux(b, time_th, zp=zp, zpsys='ab') / x0

    fint1 = M1(a * p, a * wave) * 10.**(-0.4 * CL(a * wave) * c)
    fint2 = (M0(a * p, a * wave) + x1 * M1(a * p, a * wave)) * \
        10.**(-0.4 * CL(a * wave) * c) * CL(a * wave)
    m1int = np.sum(wave * trans * fint1, axis=1) * \
        dwave / snc.constants.HC_ERG_AA
    clint = np.sum(wave * trans * fint2, axis=1) * \
        dwave / snc.constants.HC_ERG_AA

    dfdx1 = a * x0 * m1int * normfactor
    dfdc = -0.4 * np.log(10) * a * x0 * clint * normfactor
    J = np.asarray([[d1, d2, d3] for d1, d2, d3 in zip(dfdx0, dfdx1, dfdc)])
    err_th = np.sqrt(np.einsum('ki,ki->k', J, np.einsum('ij,kj->ki', cov, J)))
    return err_th


def find_filters(filter_table):
    """Find the different filter in a table.

    Parameters
    ----------
    filter_table : numpi.ndarray(str)
        Array of filters names.

    Returns
    -------
    list(str)
        List of the different filters used in the input list.

    """
    filter_list = []
    for f in filter_table:
        if f not in filter_list:
            filter_list.append(f)
    return filter_list


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
    norm_factor = pw(10, 0.4 * (zp - flux_table['zp']))
    flux_norm = flux_table['flux'] * norm_factor
    fluxerr_norm = flux_table['fluxerr'] * norm_factor
    return flux_norm, fluxerr_norm


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


def plot_lc(
        flux_table,
        zp=25.,
        mag=False,
        snc_sim_model=None,
        snc_fit_model=None,
        fit_cov=None,
        residuals=False):
    """Ploting a lightcurve flux table.

    Parameters
    ----------
    flux_table : astropy.Table
        The lightcurve to plot.
    zp : float, default = 25.
        Zeropoint at which rescale the flux.
    mag : boolean
        If True plot the magnitude.
    snc_sim_model : sncosmo.Model
        Model used to simulate the lightcurve.
    snc_fit_model : sncosmo.Model
        Model used to fit the lightcurve.
    fit_cov : numpy.ndaray(float, size=(3,3))
        sncosmo x0,x1,c covariance matrix from SALT fit.
    residuals : boolean
        If True plot fit residuals.

    Returns
    -------
    None
        Just plot the lightcurve.

    """
    bands = find_filters(flux_table['band'])
    flux_norm, fluxerr_norm = norm_flux(flux_table, zp)
    time = flux_table['time']

    t0 = flux_table.meta['sim_t0']
    z = flux_table.meta['z']
    time_th = np.linspace(t0 - 19.8 * (1 + z), t0 + 49.8 * (1 + z), 200)

    if residuals:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
    else:
        ax0 = plt.subplot(111)

    plt.xlabel('Time to peak')

    for b in bands:
        band_mask = flux_table['band'] == b
        flux_b = flux_norm[band_mask]
        fluxerr_b = fluxerr_norm[band_mask]
        time_b = time[band_mask]

        if mag:
            plt.gca().invert_yaxis()
            ax0.set_ylabel('Mag')

            # Delete < 0 pts
            flux_mask = flux_b > 0
            flux_b = flux_b[flux_mask]
            fluxerr_b = fluxerr_b[flux_mask]
            time_b = time_b[flux_mask]

            plot = -2.5 * np.log10(flux_b) + zp
            err = 2.5 / np.log(10) * 1 / flux_b * fluxerr_b

            if snc_sim_model is not None:
                plot_th = snc_sim_model.bandmag(b, 'ab', time_th)

            if snc_fit_model is not None:
                plot_fit = snc_fit_model.bandmag(b, 'ab', time_th)
                if fit_cov is not None:
                    if snc_fit_model.source.name in ('salt2', 'salt3'):
                        err_th = compute_salt_fit_error(snc_fit_model, fit_cov, b, time_th, zp)
                        err_th = 2.5 / \
                            (np.log(10) * pw(10, -0.4 * (plot_fit - zp))) * err_th
                if residuals:
                    fit_pts = snc_fit_model.bandmag(b, 'ab', time_b)
                    rsd = plot - fit_pts

        else:
            ax0.set_ylabel('Flux')
            ax0.axhline(ls='dashdot', c='black', lw=1.5)
            plot = flux_b
            err = fluxerr_b

            if snc_sim_model is not None:
                plot_th = snc_sim_model.bandflux(b, time_th, zp=zp, zpsys='ab')

            if snc_fit_model is not None:
                plot_fit = snc_fit_model.bandflux(b, time_th, zp=zp, zpsys='ab')
                if fit_cov is not None:
                    if snc_fit_model.source.name in ('salt2','salt3'):
                        err_th = compute_salt_fit_error(snc_fit_model, fit_cov, b, time_th, zp)
                if residuals:
                    fit_pts = snc_fit_model.bandflux(b, time_b, zp=zp, zpsys='ab')
                    rsd = plot - fit_pts

        p = ax0.errorbar(time_b - t0, plot, yerr=err, label=b, fmt='o', markersize=2.5)
        handles, labels = ax0.get_legend_handles_labels()

        if snc_sim_model is not None:
            ax0.plot(time_th - t0, plot_th, color=p[0].get_color())
            sim_line = Line2D([0], [0], color='k', linestyle='solid')
            sim_label = 'Sim'
            handles.append(sim_line)
            labels.append(sim_label)

        if snc_fit_model is not None:
            ax0.plot(time_th - t0, plot_fit, color=p[0].get_color(), ls='--')
            fit_line = Line2D([0], [0], color='k', linestyle='--')
            fit_label = 'Fit'
            handles.append(fit_line)
            labels.append(fit_label)

            if fit_cov is not None:
                ax0.fill_between(
                    time_th - t0,
                    plot_fit - err_th,
                    plot_fit + err_th,
                    alpha=0.5)

            if residuals:
                ax1.set_ylabel('Data - Model')
                ax1.errorbar(time_b - t0, rsd, yerr=err, fmt='o')
                ax1.axhline(0, ls='dashdot', c='black', lw=1.5)
                ax1.set_ylim(-np.max(abs(rsd)) * 2, np.max(abs(rsd)) * 2)
                ax1.plot(time_th - t0, err_th, ls='--', color=p[0].get_color())
                ax1.plot(time_th - t0, -err_th, ls='--', color=p[0].get_color())

    ax0.legend(handles=handles, labels=labels)
    plt.xlim(snc_sim_model.mintime() - t0, snc_sim_model.maxtime() - t0)
    plt.subplots_adjust(hspace=.0)
    plt.show()


def plot_ra_dec(ra, dec, vpec=None, **kwarg):
    """Plot a mollweide map of ra, dec.

    Parameters
    ----------
    ra : list(float)
        Right Ascension.
    dec : type
        Declinaison.
    vpec : type
        Peculiar velocities.

    Returns
    -------
    None
        Just plot the map.

    """
    plt.figure()
    ax = plt.subplot(111, projection='mollweide')
    plt.grid()
    ax.set_axisbelow(True)
    ra = ra - 2 * np.pi * (ra > np.pi)
    if vpec is None:
        plt.scatter(ra, dec, **kwarg)
    else:
        plot = plt.scatter(ra, dec, c=vpec, vmin=-1500, vmax=1500, **kwarg)
        plt.colorbar(plot, label='$v_p$ [km/s]')

    plt.show()


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
    MName = sim_meta['MName']
    if MName in ('salt2', 'salt3'):
        fit_keys += ['x0', 'e_x0', 'mb', 'e_mb', 'x1',
                     'e_x1', 'c', 'e_c', 'cov_x0_x1', 'cov_x0_c',
                     'cov_mb_x1', 'cov_mb_c', 'cov_x1_c']

    for k in fit_keys:
        data[k] = []

    for i in sim_lc_meta['sn_id']:
        if fit_res[i] != np.nan:
            par = fit_res[i]['parameters']
            data['t0'].append(par[1])
            data['e_t0'].append(np.sqrt(fit_res[i]['covariance'][0, 0]))

            if MName in ('salt2', 'salt3'):
                par_cov = fit_res[i]['covariance'][1:, 1:]
                mb_cov = cov_x0_to_mb(par[2], par_cov)
                data['x0'].append(par[2])
                data['e_x0'].append(np.sqrt(par_cov[0, 0]))
                data['mb'].append(x0_to_mB(par[2]))
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

            data['chi2'].append(fit_res[i]['chisq'])
            data['ndof'].append(fit_res[i]['ndof'])
        else:
            for k in fit_keys:
                data[k].append(np.nan)

    table = Table(data)

    hdu = fits.table_to_hdu(table)
    hdu_list = fits.HDUList([fits.PrimaryHDU(header=fits.Header(sim_meta)), hdu])
    hdu_list.writeto(directory, overwrite=True)
    print(f'Fit result output file : {directory}')
