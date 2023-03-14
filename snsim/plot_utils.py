"""Contains plot functions."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from . import utils as ut
from . import salt_utils as salt_ut
from . import nb_fun as nbf


def param_text_box(text_ax, model_name, sim_par=None, fit_par=None, pos=[0.01, 0.4]):
    """Add a text legend with model parameters to the plot.

    Parameters
    ----------
    text_ax : matplotlib.axes
        Axes where place the text.
    model_name : str
        The name of the sn model that is used.
    sim_par : list(float)
        The parameters of the model.
    fit_par : list(tuple(float,float))
        The fitted parameters and errors.

    """
    par_dic_salt = {'salt': {'t0': ('$t_0$', '.2f'), 'x0': ('$x_0$', '.2e'),
                    'mb': ('$m_b$', '.2f'), 'x1': ('$x_1$', '.2f'), 'c': ('$c$', '.3f')},
                    'mw_': {'mw_r_v': ('$R_v$', '.2f'), 'mw_ebv': ('E(B-V)', '.3f')}}

    par = {}
    for model in model_name:
        if model == 'salt':
            par = {**par, **par_dic_salt[model]}
        elif model != 'salt':
            par_dic_no_salt = {model: {'t0': ('$t_0$', '.2f'), 'amplitude': ('$amplitude$', '.2e'),'mb': ('$m_b$', '.2f')},
                               'mw_': {'mw_r_v': ('$R_v$', '.2f'), 'mw_ebv': ('E(B-V)', '.3f')}}
            par = {**par,**par_dic_no_salt[model]}

    str = ''
    if sim_par is not None:
        str += 'SIMULATED PARAMETERS : \n    '
        for k in par.keys():
            if k in sim_par:
                str += f"{par[k][0]} = {sim_par[k]:{par[k][1]}}  "
        str += '\n\n'

    if fit_par is not None:
        str += 'FITTED PARAMETERS : \n    '
        for k in par.keys():
            if k in fit_par:
                if isinstance(fit_par[k], (int, np.integer, float, np.floating)):
                    str += f"{par[k][0]} = {fit_par[k]:{par[k][1]}}  "
                else:
                    str += f"{par[k][0]} = {fit_par[k][0]:{par[k][1]}} $\pm$ {fit_par[k][1]:{par[k][1]}}  "

    prop = dict(boxstyle='round,pad=1', facecolor='navajowhite', alpha=0.3)
    text_ax.axis('off')
    text_ax.text(pos[0], pos[1], str, transform=text_ax.transAxes, fontsize=10, bbox=prop)


def plot_lc(
        flux_table,
        meta,
        zp=25.,
        mag=False,
        Jy=False,
        snc_sim_model=None,
        snc_fit_model=None,
        fit_cov=None,
        residuals=False,
        bandcol=None,
        set_main=None,
        set_res=None,
        flux_limit=None,
        phase_limit=[-21,51],
        mtpstyle='seaborn-deep',
        dpi=100,
        savefig=False, savepath='LC', saveformat='png'):
    """Ploting a lightcurve flux table.

    Parameters
    ----------
    flux_table : pandas.Series
        The lightcurve to plot.
    meta : dict.
        The lightcurve meta data.
    zp : float, default = 25.
        Zeropoint at which rescale the flux.
    mag : bool
        If True plot the magnitude.
    Jy: bool
        If True plot the flux in Jansky.
    snc_sim_model : sncosmo.Model
        Model used to simulate the lightcurve.
    snc_fit_model : sncosmo.Model
        Model used to fit the lightcurve.
    fit_cov : numpy.ndarray(float, size=(4, 4))
        sncosmo t0, x0, x1, c covariance matrix from SALT fit.
    residuals : bool
        If True plot fit residuals.
    bandcol : dict
        Give the color to use for each band.
    set_main : dict
        Pass matplotlib options as ax.set(**set_main).
    set_res : dict
        Same as pass main but for the residuals axis.
    mtpstyle : str
        The matplotlib style to use.
    dpi : int
        The dpi to use.
    savefig : bool
        If true save the figure rather than plot.
    savepath : str
        The path to save the figure.
    saveformat : str
        Format to use ex : png, pdf.

    """

    figsize = (15, 8)

    bands = flux_table['band'].unique()

    flux_norm, fluxerr_norm = ut.norm_flux(flux_table, zp)

    time = flux_table['time']

    t0 = meta['sim_t0']
    z = meta['zobs']

    time_th = np.linspace(t0 + ((phase_limit[0]+1.2) * (1 + z)), t0 + ((phase_limit[1]-1.2) * (1 + z)), 200)
    with plt.style.context(mtpstyle):

        fig = plt.figure(figsize=figsize, dpi=dpi)

        ###################
        # INIT THE FIGURE #
        ###################

        if residuals:
            gs = gridspec.GridSpec(3, 1, height_ratios=np.array([0.5, 2, 1]), figure=fig)
            text_ax = fig.add_subplot(gs[0])
            ax0 = fig.add_subplot(gs[1])
            ax1 = fig.add_subplot(gs[2], sharex=ax0)
            ax1_y_lim = []
        elif snc_sim_model is None and (snc_fit_model is None or fit_cov is None):
            gs = gridspec.GridSpec(1, 1, height_ratios=[1])
            ax0 = fig.add_subplot(gs[0])
        else:
            gs = gridspec.GridSpec(2, 1, height_ratios=np.array([0.3, 2]))
            text_ax = fig.add_subplot(gs[0])
            ax0 = fig.add_subplot(gs[1])

        if bandcol is None:
            bandcol = {}
            ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for i, b in enumerate(bands):
                bandcol[b] = ccycle[i]

        if set_main is None:
            set_main = {}
        if set_res is None:
            set_res = {}

        fig.suptitle(f'SN at redshift z : {z:.5f} and peak at time t$_0$ : {t0:.2f} MJD',
                    fontsize='xx-large')

        plt.xlim(phase_limit[0] * (1 + z), phase_limit[1] * (1 + z))

        if flux_limit is not None:
            ax0.set_ylim(flux_limit[0], flux_limit[1])

        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)
        ax0.spines['bottom'].set_linewidth(2)
        ax0.spines['left'].set_linewidth(2)
        ax0.xaxis.set_tick_params(width=2)
        ax0.yaxis.set_tick_params(width=2)
        ax0.set_xlabel('Phase [days]', fontsize='x-large')

        ################
        # PLOT SECTION #
        ################

        for b in bands:
            band_mask = flux_table['band'] == b
            flux_b = flux_norm[band_mask]
            fluxerr_b = fluxerr_norm[band_mask]
            time_b = time[band_mask]

            if mag:
                ax0.invert_yaxis()
                ax0.set_ylabel('Mag', fontsize='x-large')

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
                            err_th = salt_ut.compute_salt_fit_error(snc_fit_model,
                                                                    fit_cov[1:, 1:],
                                                                    b, time_th, zp)
                            err_th = 2.5 / \
                                (np.log(10) * 10**(-0.4 * (plot_fit - zp))) * err_th
                    if residuals:
                        fit_pts = snc_fit_model.bandmag(b, 'ab', time_b)
                        rsd = plot - fit_pts

            else:
                if Jy:
                    ax0.set_ylabel('Flux [$\mu$Jy]', fontsize='x-large')
                    norm = ut.flux_to_Jansky(zp, b)
                else:
                    ax0.set_ylabel(f'Flux (ZP = {zp})', fontsize='x-large')
                    norm = 1.0

                ax0.axhline(ls='dashdot', c='black', lw=1.5)
                plot = flux_b * norm
                err = fluxerr_b * norm

                if snc_sim_model is not None:
                    plot_th = snc_sim_model.bandflux(b, time_th, zp=zp, zpsys='ab') * norm

                if snc_fit_model is not None:
                    plot_fit = snc_fit_model.bandflux(
                        b, time_th, zp=zp, zpsys='ab') * norm
                    if fit_cov is not None:
                        if snc_fit_model.source.name in ('salt2', 'salt3'):
                            err_th = salt_ut.compute_salt_fit_error(snc_fit_model, fit_cov[1:, 1:],
                                                                    b, time_th, zp) * norm

                    if residuals:
                        fit_pts = snc_fit_model.bandflux(b, time_b, zp=zp, zpsys='ab') * norm
                        rsd = plot - fit_pts

            ax0.errorbar(time_b - t0, plot, yerr=err,
                        label=b, fmt='o', ms=5, lw=1.5, color=bandcol[b])

            handles, labels = ax0.get_legend_handles_labels()

            if snc_sim_model is not None:
                ax0.plot(time_th - t0, plot_th, color=bandcol[b])
                sim_line = Line2D([0], [0], color='k', linestyle='solid')
                sim_label = 'Sim'
                handles.append(sim_line)
                labels.append(sim_label)

            if snc_fit_model is not None:
                fit_line = Line2D([0], [0], color='k', linestyle='--')
                fit_label = 'Fit'
                handles.append(fit_line)
                labels.append(fit_label)
                ax0.plot(time_th - t0, plot_fit, color=bandcol[b], ls='--')

                if fit_cov is not None:
                    ax0.fill_between(
                        time_th - t0,
                        plot_fit - err_th,
                        plot_fit + err_th,
                        alpha=0.3, lw=0.0,
                        color=bandcol[b])

                if residuals:
                    ax1.set_ylabel('Data - Model', fontsize='x-large')
                    ax1.errorbar(time_b - t0, rsd, yerr=err, fmt='o', color=bandcol[b], ms=5)
                    ax1.axhline(0, ls='dashdot', c='black', lw=1.5)
                    ax1_y_lim.append(3 * np.std(rsd))
                    ax1.plot(time_th - t0, err_th, ls='--', color=bandcol[b])
                    ax1.plot(time_th - t0, -err_th, ls='--', color=bandcol[b])
                    for axis in ['top', 'bottom', 'left', 'right']:
                        ax1.spines[axis].set_linewidth(2)
                    ax1.xaxis.set_tick_params(width=2)
                    ax1.yaxis.set_tick_params(width=2)
                    ax1.set_xlabel('Phase [days]', fontsize='x-large')
                    ax1.set(**set_res)
                    plt.setp(ax0.get_xticklabels(), visible=False)
                    ax0.set_xlabel('')

        ax0.legend(handles=handles, labels=labels, fontsize='x-large')

        sim_par = None
        fit_par = None
        model_name = []

        if snc_sim_model is not None:
            model_name.append(snc_sim_model.source.name[:-1])
            sim_par = {snc_sim_model.param_names[i]: snc_sim_model.parameters[i]
                    for i in range(len(snc_sim_model.param_names))}
            if snc_sim_model.source.name[:-1] == 'salt':
                sim_par['mb'] = snc_sim_model.source_peakmag('bessellb', 'ab')
            elif snc_sim_model.source.name[:-1] != 'salt':		
                sim_par['mb'] = snc_sim_model.source_peakmag('bessellb', 'ab')

            if 'mw_' in snc_sim_model.effect_names:
                model_name.append('mw_')

        if snc_fit_model is not None and fit_cov is not None:
            model_name.append(snc_fit_model.source.name[:-1])
            fit_par = {}
            for i in range(1, len(snc_fit_model.param_names)):
                if 'mw_' not in snc_fit_model.param_names[i]:
                    fit_par[snc_fit_model.param_names[i]] = (snc_fit_model.parameters[i], np.sqrt(fit_cov[i-1, i-1]))
                else:
                    fit_par[snc_fit_model.param_names[i]] = snc_fit_model.parameters[i]
            if snc_fit_model.source.name[:-1] == 'salt':
                fit_par['mb'] = (snc_fit_model.source_peakmag('bessellb', 'ab'),
                                np.sqrt(salt_ut.cov_x0_to_mb(snc_fit_model.parameters[2], fit_cov[1:, 1:])[0, 0]))

            if 'mw_' in snc_sim_model.effect_names:
                model_name.append('mw_')

        if model_name != []:
            param_text_box(text_ax, model_name=model_name, sim_par=sim_par, fit_par=fit_par)

        plt.subplots_adjust(hspace=.08)
        ax0.set(**set_main)

    if savefig:
        plt.savefig(f'{savepath}.{saveformat}', dpi=dpi, format=saveformat)
    else:
        plt.show()


def plot_ra_dec(ra, dec, vpec=None, field_list=None, field_dic=None, field_size=None, **kwarg):
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
    ax.set_axisbelow(True)
    plt.grid()

    ra = ra - 2 * np.pi * (ra > np.pi)

    if vpec is None:
        plt.scatter(ra, dec, s=10, **kwarg)
    else:
        plot = plt.scatter(ra, dec, c=vpec, vmin=-1500, vmax=1500, s=10, **kwarg)
        plt.colorbar(plot, label='$v_p$ [km/s]')

    if field_list is not None and field_dic is not None and field_size is not None:
        ra_edges = np.array([field_size[0] / 2,
                             field_size[0] / 2,
                             -field_size[0] / 2,
                             -field_size[0] / 2])

        dec_edges = np.array([field_size[1] / 2,
                              -field_size[1] / 2,
                              -field_size[1] / 2,
                              field_size[1] / 2])

        vec = np.array([np.cos(ra_edges) * np.cos(dec_edges),
                        np.sin(ra_edges) * np.cos(dec_edges),
                        np.sin(dec_edges)]).T

        for ID in field_list:
            # if ID < 880:
            ra = field_dic[ID]['ra']
            dec = field_dic[ID]['dec']
            new_coord = [nbf.R_base(
                ra, -dec, v, to_field_frame=False) for v in vec]
            new_radec = [[np.arctan2(x[1], x[0]), np.arcsin(x[2])] for x in new_coord]

            if new_radec[3][0] > new_radec[0][0]:
                if new_radec[3][0] * new_radec[2][0] > 0:
                    x1 = [-np.pi, new_radec[0][0], new_radec[0][0], -np.pi]
                    y1 = [new_radec[0][1], new_radec[0][1],
                          new_radec[1][1], new_radec[1][1]]
                    x2 = [np.pi, new_radec[2][0], new_radec[2][0], np.pi]
                    y2 = [new_radec[2][1], new_radec[2][1],
                          new_radec[3][1], new_radec[3][1]]
                    ax.plot(x1, y1, ls='--', color='blue', lw=1, zorder=2)
                    ax.plot(x2, y2, ls='--', color='blue', lw=1, zorder=2)
                else:
                    if new_radec[2][0] < 0:
                        new_radec[3][0] = -np.pi
                        plt.gca().add_patch(Polygon(new_radec,
                                                    fill=False,
                                                    ls='--',
                                                    color='blue',
                                                    lw=1,
                                                    zorder=2))

            else:
                plt.gca().add_patch(Polygon(new_radec,
                                            fill=False,
                                            ls='--',
                                            color='blue',
                                            lw=1,
                                            zorder=2))
    plt.show()
