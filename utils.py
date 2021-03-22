from astropy import constants as cst
import sncosmo as snc
import numpy as np
from numpy import power as pw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sn_sim_print = '     _______..__   __.         _______. __  .___  ___. \n'
sn_sim_print += '    /       ||  \\ |  |        /       ||  | |   \\/   | \n'
sn_sim_print += '   |   (----`|   \\|  |       |   (----`|  | |  \\  /  | \n'
sn_sim_print += '    \\   \\    |  . `  |        \\   \\    |  | |  |\\/|  | \n'
sn_sim_print += '.----)   |   |  |\\   |    .----)   |   |  | |  |  |  | \n'
sn_sim_print += '|_______/    |__| \\__|    |_______/    |__| |__|  |__| \n'

sep = '###############################################'

c_light_kms = cst.c.to('km/s').value
# just an offset -> set_peakmag(mb=0,'bessellb', 'ab') ->
# offset=2.5*log10(get_x0) change with magsys
snc_mag_offset = 10.5020699


def x0_to_mB(x0):
    '''Convert x0 to mB'''
    return -2.5 * np.log10(x0) + snc_mag_offset


def mB_to_x0(mB):
    '''Convert mB to x0'''
    return pw(10, -0.4 * (mB - snc_mag_offset))


def cov_x0_to_mb(x0, cov):
    J = np.array([[-2.5 / np.log(10) * 1 / x0, 0, 0], [0, 1, 0], [0, 0, 1]])
    new_cov = J @ cov @ J.T
    return new_cov


def box_output(sep, line):
    '''Use for plotting simulation output'''
    l = len(sep) - len(line) - 2
    space1 = ' ' * (l // 2)
    space2 = ' ' * (l // 2 + l % 2)
    return '#' + space1 + line + space2 + '#'


def snc_fit(lc, model):
    '''Fit the given lc with the given SALT2 model
       Free parameters are : - The SN peak magnitude in B-band t0
                             - The normalisation factor x0 (<=> mB)
                             - The stretch parameter x1
                             - The color parameter c
    '''
    return snc.fit_lc(lc, model, ['t0', 'x0', 'x1', 'c'], modelcov=True)


def compute_fit_error(fit_model, cov, band, flux_th, time_th, zp, magsys='ab'):
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
    COV = cov
    b = snc.get_bandpass(band)
    wave, dwave = snc.utils.integration_grid(
        b.minwave(), b.maxwave(), snc.constants.MODEL_BANDFLUX_SPACING)
    trans = b(wave)
    ms = snc.get_magsystem(magsys)
    zpms = ms.zpbandflux(b)
    normfactor = 10**(0.4 * zp) / zpms
    err_th = []
    for t, f in zip(time_th, flux_th):
        p = time_th - t0
        dfdx0 = f / x0
        fint1 = fit_model.source._model['M1'](
            a * p, a * wave)[0] * 10.**(-0.4 * fit_model.source._colorlaw(a * wave) * c)
        fint2 = (fit_model.source._model['M0'](a * p, a * wave)[0] + x1 * fit_model.source._model['M1'](a * p, a * wave)[
                 0]) * 10.**(-0.4 * fit_model.source._colorlaw(a * wave) * c) * fit_model.source._colorlaw(a * wave)
        m1int = np.sum(wave * trans * fint1, axis=0) * \
            dwave / snc.constants.HC_ERG_AA
        clint = np.sum(wave * trans * fint2, axis=0) * \
            dwave / snc.constants.HC_ERG_AA
        dfdx1 = a * x0 * m1int * normfactor
        dfdc = -0.4 * np.log(10) * a * x0 * clint * normfactor
        J = np.asarray([dfdx0, dfdx1, dfdc], dtype=float)
        err = np.sqrt(J.T @ cov @ J)
        err_th.append(err)
    err_th = np.asarray(err_th)
    return err_th


def plot_lc(
        flux_table,
        zp=25.,
        mag=False,
        sim_model=None,
        fit_model=None,
        fit_cov=None,
        residuals=False):
    '''General plot function
       Options : - zp = float, use the normalisation zero point that you want (default: 25.)
                 - mag = boolean, plot magnitude (default = False)
    '''

    bands = find_filters(flux_table['band'])
    flux_norm, fluxerr_norm = norm_flux(flux_table, zp)
    time = flux_table['time']

    t0 = flux_table.meta['t0']
    z = flux_table.meta['z']

    time_th = np.linspace(t0 - 19.8 * (1 + z), t0 + 49.8 * (1 + z), 500)

    fig = plt.figure()
    if residuals:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
    else:
        ax0 = plt.subplot(111)

    if sim_model is not None:
        x0 = flux_table.meta['x0']
        mb = x0_to_mB(flux_table.meta['x0'])
        x1 = flux_table.meta['x1']
        c = flux_table.meta['c']

        sim_model.set(z=z, c=c, t0=t0, x0=x0, x1=x1)

        title = f'z = {z:.3f} $m_B$ = {mb:.3f} $x_1$ = {x1:.3f} $c$ = {c:.4f}'
        ax0.set_title(title)

    plt.xlabel('Time to peak')
    ylim = 0
    for b in bands:
        band_mask = flux_table['band'] == b
        flux_b = flux_norm[band_mask]
        fluxerr_b = fluxerr_norm[band_mask]
        time_b = time[band_mask]

        if mag:
            plt.gca().invert_yaxis()
            ax0.set_ylabel('Mag')
            # Delete < 0 pts
            flux_b, fluxerr_b, time_b = flux_b[flux_b >
                                               0], fluxerr_b[flux_b > 0], time_b[flux_b > 0]
            plot = -2.5 * np.log10(flux_b) + zp
            plt.ylim(np.max(plot) + 3, np.min(plot) - 3)
            err = 2.5 / np.log(10) * 1 / flux_b * fluxerr_b
            if sim_model is not None:
                plot_th = sim_model.bandmag(b, 'ab', time_th)
            if fit_model is not None:
                plot_fit = fit_model.bandmag(b, 'ab', time_th)
                if fit_cov is not None:
                    err_th = compute_fit_error(fit_model, fit_cov, b, pw(
                        10, -0.4 * (plot_fit - zp)), time_th, zp)
                    err_th = 2.5 / \
                        (np.log(10) * pw(10, -0.4 * (plot_fit - zp))) * err_th
                if residuals:
                    fit_pts = fit_model.bandmag(b, 'ab', time_b)
                    rsd = plot - fit_pts

        else:
            ax0.set_ylabel('Flux')
            ax0.axhline(ls='dashdot', c='black', lw=1.5)
            plot = flux_b
            err = fluxerr_b
            if sim_model is not None:
                plot_th = sim_model.bandflux(b, time_th, zp=zp, zpsys='ab')
                ylim = ylim + (np.max(plot_th) - ylim) * \
                    (np.max(plot_th) > ylim)
            if fit_model is not None:
                plot_fit = fit_model.bandflux(b, time_th, zp=zp, zpsys='ab')
                if fit_cov is not None:
                    err_th = compute_fit_error(
                        fit_model, fit_cov, b, plot_fit, time_th, zp)
                if residuals:
                    fit_pts = fit_model.bandflux(b, time_b, zp=zp, zpsys='ab')
                    rsd = plot - fit_pts

        p = ax0.errorbar(
            time_b - t0,
            plot,
            yerr=err,
            label=b,
            fmt='o',
            markersize=2.5)
        if sim_model is not None:
            ax0.plot(time_th - t0, plot_th, color=p[0].get_color())
        if fit_model is not None:
            ax0.plot(time_th - t0, plot_fit, color=p[0].get_color(), ls='--')
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
                ax1.plot(time_th - t0, -err_th, ls='--',
                         color=p[0].get_color())

    # plt.ylim(-np.max(ylim)*0.1,np.max(ylim)*1.1)
    handles, labels = ax0.get_legend_handles_labels()
    if sim_model is not None:
        sim_line = Line2D([0], [0], color='k', linestyle='solid')
        sim_label = 'Sim'
        handles.append(sim_line)
        labels.append(sim_label)

    if fit_model is not None:
        fit_line = Line2D([0], [0], color='k', linestyle='--')
        fit_label = 'Fit'
        handles.append(fit_line)
        labels.append(fit_label)

    ax0.legend(handles=handles, labels=labels)
    plt.subplots_adjust(hspace=.0)
    plt.show()
    return


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


def add_filter(path):  # Not implemented yet for later purpose
    input_name = {}
    for band in bands:
        table = np.loadtxt(band[1])
        name = band[0]
        band = snc.Bandpass(wavelength, transmission, name=name)
        try:
            snc.register(band)
        except (Exception):
            band.name += '_temp'
            snc.register(band, force=True)
            input_name[band[0]] = band.name
    if input_name == {}:
        return None
    else:
        return input_name
