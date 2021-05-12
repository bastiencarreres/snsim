from astropy import constants as cst
import sncosmo as snc
import numpy as np
from astropy.table import Table
from astropy.io import fits
from numpy import power as pw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from . import nb_fun as nbf
from .constants import SNC_MAG_OFFSET_AB

def x0_to_mB(x0):
    '''Convert x0 to mB'''
    return -2.5 * np.log10(x0) + SNC_MAG_OFFSET_AB


def mB_to_x0(mB):
    '''Convert mB to x0'''
    return pw(10, -0.4 * (mB - SNC_MAG_OFFSET_AB))


def cov_x0_to_mb(x0, cov):
    J = np.array([[-2.5 / np.log(10) * 1 / x0, 0, 0], [0, 1, 0], [0, 0, 1]])
    new_cov = J @ cov @ J.T
    return new_cov

def init_sn_model(name, model_dir):
    if name == 'salt2':
        return snc.Model(source=snc.SALT2Source(model_dir, name='salt2'))
    if name == 'salt3':
        return snc.Model(source=snc.SALT3Source(model_dir, name='salt3'))

def box_output(sep, line):
    '''Use for plotting simulation output'''
    l = len(sep) - len(line) - 2
    space1 = ' ' * (l // 2)
    space2 = ' ' * (l // 2 + l % 2)
    return '#' + space1 + line + space2 + '#'

def snc_fitter(lc,fit_model,fit_par):
    try:
            res = snc.fit_lc(lc, fit_model, fit_par,  modelcov=True)[0]
    except BaseException:
            res = np.nan
    return res

def compute_salt_fit_error(fit_model, cov, band, flux_th, time_th, zp, magsys='ab'):
    '''Compute theorical fluxerr from fit err = sqrt(COV)
    where COV = J**T * COV(x0,x1,c) * J with J = (dF/dx0, dF/dx1, dF/dc) the jacobian.
    According to Fnorm = x0/(1+z) * int_\\lambda (M0(lambda_s,p)+x1*M1(lambda_s,p))*10**(-0.4*c*CL(lambda_s)) * T_b(lambda) * lambda/hc dlambda * norm_factor
    where norm_factor = 10**(0.4*ZP_norm)/ZP_magsys. We found :
    dF/dx0 = F/x0
    dF/dx1 = x0/(1+z) * int_lambda M1(lambda_s,p))*10**(-0.4*c*CL(lambda_s)) * T_b(lambda) * lambda/hc dlambda * norm_factor
    dF/dc  =  -0.4*ln(10)*x0/(1+z) * int_\\lambda (M0(lambda_s,p)+x1*M1(lambda_s,p))*CL(lambda_s)*10**(-0.4*c*CL(lambda_s)) * T_b(lambda) * lambda/hc dlambda * norm_factor
    '''
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

    dfdx0 = flux_th / x0

    fint1 = M1(a * p, a * wave) * 10.**(-0.4 * CL(a * wave) * c)
    fint2 = (M0(a * p, a * wave) + x1 * M1(a * p, a * wave)) * 10.**(-0.4 * CL(a * wave) * c) * CL(a * wave)
    m1int = np.sum(wave * trans * fint1, axis=1) * \
            dwave / snc.constants.HC_ERG_AA
    clint = np.sum(wave * trans * fint2, axis=1) * \
            dwave / snc.constants.HC_ERG_AA

    dfdx1 = a * x0 * m1int * normfactor
    dfdc = -0.4 * np.log(10) * a * x0 * clint * normfactor
    J = np.asarray([[d1, d2, d3] for d1,d2,d3 in zip(dfdx0, dfdx1, dfdc)])
    err_th = np.sqrt(np.einsum('ki,ki->k',J, np.einsum('ij,kj->ki', cov, J)))
    return err_th


def find_filters(filter_table):
    '''Take a list of obs filter and return the name of the different filters'''
    filter_list = []
    for f in filter_table:
        if f not in filter_list:
            filter_list.append(f)
    return filter_list


def norm_flux(flux_table, zp):
    '''Taken from sncosmo -> set the flux to the same zero-point'''
    norm_factor = pw(10, 0.4 * (zp - flux_table['zp']))
    flux_norm = flux_table['flux'] * norm_factor
    fluxerr_norm = flux_table['fluxerr'] * norm_factor
    return flux_norm, fluxerr_norm

def sine_interp(x_new, fun_x, fun_y):
    if len(fun_x) != len(fun_y):
        raise ValueError('x and y must have the same len')

    if (x_new > fun_x[-1]) or (x_new < fun_x[0]):
        raise ValueError('x_new is out of range of fun_x')

    inf_sel = x_new >= fun_x[:-1]
    sup_sel = x_new < fun_x[1:]
    if inf_sel.all():
        idx_inf = -2
    elif sup_sel.all():
        idx_inf = 0
    else:
        idx_inf=np.where(inf_sel*sup_sel)[0][0]

    x_inf = fun_x[idx_inf]
    x_sup = fun_x[idx_inf+1]
    Value_inf = fun_y[idx_inf]
    Value_sup = fun_y[idx_inf+1]
    sin_interp = np.sin(np.pi*(x_new-0.5*(x_inf+x_sup))/(x_sup-x_inf))

    return 0.5*(Value_sup+Value_inf)+0.5*(Value_sup-Value_inf)*sin_interp

def change_sph_frame(ra,dec,ra_frame,dec_frame):
    if isinstance(ra_frame,float):
        ra_frame=np.array([ra_frame])
        dec_frame=np.array([dec_frame])
    vec=np.array([np.cos(ra)*np.cos(dec),np.sin(ra)*np.cos(dec),np.sin(dec)])
    new_ra,new_dec = nbf.new_coord_on_fields(ra_frame,dec_frame,vec)
    return new_ra, new_dec

def plot_lc(flux_table, zp=25., mag=False, snc_sim_model=None, snc_fit_model=None, fit_cov=None, residuals=False):

    '''General plot function
       Options : - zp = float, use the normalisation zero point that you want (default: 25.)
                 - mag = boolean, plot magnitude (default = False)
    '''

    bands = find_filters(flux_table['band'])
    flux_norm, fluxerr_norm = norm_flux(flux_table, zp)
    time = flux_table['time']

    t0 = flux_table.meta['sim_t0']
    z = flux_table.meta['z']
    time_th = np.linspace(t0 - 19.8 * (1 + z), t0 + 49.8 * (1 + z), 200)

    fig = plt.figure()
    plt.xlabel('Time to peak')

    if residuals:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
    else:
        ax0 = plt.subplot(111)

    for b in bands:
        band_mask = flux_table['band'] == b
        flux_b = flux_norm[band_mask]
        fluxerr_b = fluxerr_norm[band_mask]
        time_b = time[band_mask]

        if mag:
            plt.gca().invert_yaxis()
            ax0.set_ylabel('Mag')

            # Delete < 0 pts
            flux_mask = flux_b >0
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
                    if snc_fit_model.source.name == 'salt2' or snc_fit_model.source.name == 'salt3':
                        err_th = compute_salt_fit_error(snc_fit_model, fit_cov, b, pw(
                            10, -0.4 * (plot_fit - zp)), time_th, zp)
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
                plot_th = snc_sim_model.bandflux(b, time_th, zp = zp, zpsys = 'ab')

            if snc_fit_model is not None:
                plot_fit = snc_fit_model.bandflux(b, time_th, zp=zp, zpsys = 'ab')
                if fit_cov is not None:
                    if snc_fit_model.source.name == 'salt2' or fit_model.source.name == 'salt3':
                        err_th = compute_salt_fit_error(snc_fit_model, fit_cov, b, plot_fit, time_th, zp)
                if residuals:
                    fit_pts = snc_fit_model.bandflux(b, time_b, zp=zp, zpsys='ab')
                    rsd = plot - fit_pts

        p = ax0.errorbar(time_b - t0,plot,yerr=err, label=b, fmt='o',markersize= 2.5)
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
    plt.xlim(snc_sim_model.mintime()-t0,snc_sim_model.maxtime()-t0)
    plt.subplots_adjust(hspace=.0)
    plt.show()
    return
