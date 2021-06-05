"""Contains plots functions"""

import numpy as np
import snsim.utils as ut
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import snsim.SALT_utils as salt_ut


def plt_maximize():
    """Enable full screen.

    Notes
    -----
    Come from https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python/22418354#22418354
    """
    # See discussion: https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    backend = plt.get_backend()
    cfm = plt.get_current_fig_manager()
    if backend == "wxAgg":
        cfm.frame.Maximize(True)
    elif backend == "TkAgg":
        if system() == "win32":
            cfm.window.state('zoomed')  # This is windows only
        else:
            cfm.resize(*cfm.window.maxsize())
    elif backend == 'QT4Agg' or backend == 'QT5Agg':
        cfm.window.showMaximized()
    elif callable(getattr(cfm, "full_screen_toggle", None)):
        if not getattr(cfm, "flag_is_max", None):
            cfm.full_screen_toggle()
            cfm.flag_is_max = True
    else:
        raise RuntimeError("plt_maximize() is not implemented for current backend:", backend)

def param_text_box(text_ax, model_name, sim_par=None, fit_par=None):
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
    par_dic = { 'salt' : [('t0','.2f'), ('x0','.2e'), ('mb','.2f'), ('x1','.2f'), ('c','.3f')]}
    par = par_dic[model_name]

    str_list = ['']*(len(par)+1)
    if sim_par is not None:
        str_list[0] += 'SIMULATED PARAMETERS :@'
    if fit_par is not None:
        str_list[0] += 'FITTED PARAMETERS :@'

    for i, p in enumerate(par):

        if sim_par is not None:
            str_list[i+1] += f"{p[0]} = {sim_par[i]:{p[1]}}@"
        if fit_par is not None:
            str_list[i+1] += f"{p[0]} = {fit_par[i][0]:{p[1]}} $\pm$ {fit_par[i][1]:{p[1]}}@"

    final_str=""
    if str_list[0].count('@') == 2:
        len_str = []
        for i, s in enumerate(str_list):
            str_list[i] = s.split('@')
            len_str.append(len(str_list[i][0]))
        max_len = np.max(len_str)
        for i in range(len(str_list)):
            final_str += str_list[i][0] + " " * (max_len - len_str[i] + 2) + "|  "
            final_str += str_list[i][1] + "\n"
    elif str_list[0].count('@') == 1:
        for i, s in enumerate(str_list):
            final_str += str_list[i][:-1]+'\n'

    props = dict(boxstyle='round,pad=1', facecolor='navajowhite', alpha=0.5)
    text_ax.axis('off')
    text_ax.text(0.01, 0.25, final_str[:-1], transform=text_ax.transAxes, fontsize=9, bbox=props)


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
    plt.rcParams['font.family'] = 'monospace'

    bands = np.unique(flux_table['band'])
    flux_norm, fluxerr_norm = ut.norm_flux(flux_table, zp)
    time = flux_table['time']

    t0 = flux_table.meta['sim_t0']
    z = flux_table.meta['z']

    time_th = np.linspace(t0 - 19.8 * (1 + z), t0 + 49.8 * (1 + z), 200)
    fig = plt.figure(figsize=(35/2.54, 20/2.54), dpi=120)

    ###################
    # INIT THE FIGURE #
    ###################

    if residuals:
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.5, 2, 1])
        text_ax = fig.add_subplot(gs[0])
        ax0 = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[2], sharex=ax0)
        ax1_y_lim = []
    elif snc_sim_model is None and (snc_fit_model is None or fit_cov is None):
        gs = gridspec.GridSpec(1, 1, height_ratios=[1])
        ax0 = fig.add_subplot(gs[0])
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[0.5, 2])
        text_ax = fig.add_subplot(gs[0])
        ax0 = fig.add_subplot(gs[1])

    fig.suptitle(f'SN at redshift z : {z:.5f} and peak at time t$_0$ : {t0:.2f} MJD',
                fontsize='xx-large')
    plt.xlabel('Time relative to peak', fontsize='x-large')

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
                                                                fit_cov[1:,1:],
                                                                b, time_th, zp)
                        err_th = 2.5 / \
                            (np.log(10) * 10**(-0.4 * (plot_fit - zp))) * err_th
                if residuals:
                    fit_pts = snc_fit_model.bandmag(b, 'ab', time_b)
                    rsd = plot - fit_pts

        else:
            ax0.set_ylabel(f'Flux (ZP = {zp})', fontsize='x-large')
            ax0.axhline(ls='dashdot', c='black', lw=1.5)
            plot = flux_b
            err = fluxerr_b

            if snc_sim_model is not None:
                plot_th = snc_sim_model.bandflux(b, time_th, zp=zp, zpsys='ab')

            if snc_fit_model is not None:
                plot_fit = snc_fit_model.bandflux(b, time_th, zp=zp, zpsys='ab')
                if fit_cov is not None:
                    if snc_fit_model.source.name in ('salt2','salt3'):
                        err_th = salt_ut.compute_salt_fit_error(snc_fit_model, fit_cov[1:,1:], b, time_th, zp)
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
            fit_line = Line2D([0], [0], color='k', linestyle='--')
            fit_label = 'Fit'
            handles.append(fit_line)
            labels.append(fit_label)
            ax0.plot(time_th - t0, plot_fit, color=p[0].get_color(), ls='--')

            if fit_cov is not None:
                ax0.fill_between(
                    time_th - t0,
                    plot_fit - err_th,
                    plot_fit + err_th,
                    alpha=0.5)

            if residuals:
                ax1.set_ylabel('Data - Model', fontsize='x-large')
                ax1.errorbar(time_b - t0, rsd, yerr=err, fmt='o')
                ax1.axhline(0, ls='dashdot', c='black', lw=1.5)
                ax1_y_lim.append(3 * np.std(rsd))
                ax1.plot(time_th - t0, err_th, ls='--', color=p[0].get_color())
                ax1.plot(time_th - t0, -err_th, ls='--', color=p[0].get_color())

    ax0.legend(handles=handles, labels=labels, fontsize='x-large')

    sim_par = None
    fit_par = None
    if snc_sim_model is not None:
        plt.xlim(snc_sim_model.mintime() - t0, snc_sim_model.maxtime() - t0)
        sim_par = [flux_table.meta['sim_t0'],
                   flux_table.meta['sim_x0'],
                   flux_table.meta['sim_mb'],
                   flux_table.meta['sim_x1'],
                   flux_table.meta['sim_c']]

    elif snc_fit_model is not None:
        plt.xlim(snc_fit_model.mintime() - t0, snc_fit_model.maxtime() - t0)
    else:
        plt.xlim(np.min(time)-1-t0, np.max(time)+1-t0)

    if residuals:
        ax1.set_ylim(-np.max(ax1_y_lim), np.max(ax1_y_lim))

    if snc_fit_model is not None and fit_cov is not None:
        mb_fit = salt_ut.x0_to_mB(snc_fit_model.parameters[2])
        mb_err = np.sqrt(salt_ut.cov_x0_to_mb(snc_fit_model.parameters[2], fit_cov[1:,1:])[0,0])
        fit_par = [(snc_fit_model.parameters[1], np.sqrt(fit_cov[0,0])),
                   (snc_fit_model.parameters[2], np.sqrt(fit_cov[1,1])),
                   (mb_fit, mb_err),
                   (snc_fit_model.parameters[3], np.sqrt(fit_cov[2,2])),
                   (snc_fit_model.parameters[4], np.sqrt(fit_cov[3,3]))]

    if fit_par is not None or sim_par is not None:
        param_text_box(text_ax, model_name = 'salt', sim_par = sim_par, fit_par = fit_par)
    plt.subplots_adjust(hspace=.0)
    # try :
    #     plt_maximize()
    # except:
    #     pass
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
        plt.scatter(ra, dec, zorder=5, s=10, **kwarg)
    else:
        plot = plt.scatter(ra, dec, c=vpec, vmin=-1500, vmax=1500, s=10, zorder=5, **kwarg)
        plt.colorbar(plot, label='$v_p$ [km/s]')

    if field_list is not None and field_dic is not None and field_size is not None:
        ra_edges = np.array([field_size[0]/2,
                                 field_size[0]/2,
                                 -field_size[0]/2,
                                 -field_size[0]/2])
        dec_edges = np.array([field_size[1]/2,
                             -field_size[1]/2,
                             -field_size[1]/2,
                              field_size[1]/2])
        vec = np.array([np.cos(ra_edges) * np.cos(dec_edges),
                       np.sin(ra_edges) * np.cos(dec_edges),
                       np.sin(dec_edges)]).T

        for ID in field_list:
            #if ID < 880:
            ra = field_dic[ID]['ra']
            dec = field_dic[ID]['dec']
            new_coord = [nbf.R_base(ra,-dec,v, to_field_frame=False) for v in vec]
            new_radec = [[np.arctan2(x[1], x[0]), np.arcsin(x[2])] for x in new_coord]
            if new_radec[3][0] > new_radec[0][0]:
                if  new_radec[3][0]*new_radec[2][0] > 0:
                    x1 = [-np.pi, new_radec[0][0], new_radec[0][0], -np.pi]
                    y1 = [new_radec[0][1], new_radec[0][1], new_radec[1][1], new_radec[1][1]]
                    x2 = [np.pi, new_radec[2][0], new_radec[2][0], np.pi]
                    y2 = [new_radec[2][1], new_radec[2][1], new_radec[3][1], new_radec[3][1]]
                    ax.plot(x1,y1,ls='--', color='blue', lw=1, zorder=2)
                    ax.plot(x2,y2,ls='--', color='blue', lw=1, zorder=2)
                else:
                    if new_radec[2][0] < 0:
                        new_radec[3][0] = -np.pi
                        plt.gca().add_patch(Polygon(new_radec,
                                                    fill=False,
                                                    ls='--',
                                                    color='blue',
                                                    lw=1,
                                                    zorder=2))

            else :
                plt.gca().add_patch(Polygon(new_radec,
                                            fill=False,
                                            ls='--',
                                            color='blue',
                                            lw=1,
                                            zorder=2))
    plt.show()
