"""This module contains usefull function for the simulation."""

import numpy as np
import sncosmo as snc
import astropy.time as atime
from astropy.coordinates import SkyCoord
from astropy import cosmology as acosmo
import astropy.units as astu
from .constants import C_LIGHT_KMS


def gauss(mu, sig, x):
    """Gaussian function.

    Parameters
    ----------
    mu : float
        Mean.
    sig : float
        Sigma.
    x : float or numpy.array(float)
        Variables.

    Returns
    -------
    numpy.array(float)
        G(x).
    """
    return np.exp(-0.5 * ((x - mu) / sig)**2) / np.sqrt(2 * np.pi * sig**2)


class CustomRandom:
    """Class to generate random variable on custom dist.

    Parameters
    ----------
    pdf: lambda function
        Function that return the pdf of the vairable x.
    xmin: float
        Inferior bound of the distribution.
    xmax: float
        Superior bound of the distribution.
    ndiv: float, optional
        Number of division used to integrate the pdf.
    dx: float, optional
        Precision used to integrate the pdf.

    Notes
    -----
    If dx and ndiv are set, only dx will be used. 
    If none of the 2 is set, the default will be ndiv=1e4
    """    
    def __init__(self, pdf, xmin, xmax, ndiv=1e4, dx=None):
        """Init the CustomRandom class."""
        if dx is not None:
            n = int((xmax - xmin) / dx)
        else:
            n = ndiv
       
        self.x = np.linspace(xmin, xmax, n)
        self.dx = self.x[1] - self.x[0]
       
        # Compute pdf and renormalize
        self.pdfx = pdf(self.x)

        self.norm = np.trapz(self.pdfx, x=self.x)
        self.pdfx /= self.norm
        
        # Compute cdf and renormalize to be sure
        self.cdf = np.cumsum(self.pdfx) * self.dx
        
        self.cdf /= self.cdf[-1]
    
    def plot_pdf(self, ax=None):
        """Plot the pdf function.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Figure axis, by default None
        """        

        if ax is None:
            fig, ax = plt.subplots()
            
        ax.plot(self.x, self.pdfx)
        
    def plot_cdf(self, ax=None):
        """Plot the cdf function.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Figure axis, by default None
        """        
        if ax is None:
            fig, ax = plt.subplots()
            
        ax.plot(self.x, self.pdfx)
    
    def draw(self, n, seed=None):
        """Draw n parameters from the distribution.

        Parameters
        ----------
        n : int
            Number of parameters to draw.
        seed : int, optional
            Random seed, by default None

        Returns
        -------
        numpy.array
            Random parameters following the custom distribution.
        """        
        rand_gen = np.random.default_rng(seed)
        return np.interp(rand_gen.random(n), self.cdf, self.x)


def set_cosmo(cosmo_dic):
    """Load an astropy cosmological model.

    Parameters
    ----------
    cosmo_dic : dict
        A dict containing cosmology parameters.

    Returns
    -------
    astropy.cosmology.object
        An astropy cosmological model.

    """
    astropy_mod = list(map(lambda x: x.lower(), acosmo.available))
    if 'name' in cosmo_dic.keys():
        name = cosmo_dic['name'].lower()
        if name in astropy_mod:
            if name == 'planck18':
                return acosmo.Planck18
            elif name == 'planck18_arxiv_v2':
                return acosmo.Planck18_arXiv_v2
            elif name == 'planck15':
                return acosmo.Planck15
            elif name == 'planck13':
                return acosmo.Planck13
            elif name == 'wmap9':
                return acosmo.WMAP9
            elif name == 'wmap7':
                return acosmo.WMAP7
            elif name == 'wmap5':
                return acosmo.WMAP5
        else:
            raise ValueError(f'Available model are {astropy_mod}')
    else:
        if 'Ode0' not in cosmo_dic.keys():
            if 'Ok0' in cosmo_dic.keys():
                Ok0 = cosmo_dic['Ok0']
                cosmo_dic.pop('Ok0')
            else:
                Ok0 = 0.
            cosmo_dic['Ode0'] = 1 - cosmo_dic['Om0'] - Ok0
        return acosmo.w0waCDM(**cosmo_dic)


def scale_M0_jla(h):
    """Compute a value of M0 corresponding to JLA results.

    Parameters
    ----------
    h : float
        The H0 / 100 constant to scale M0.

    Returns
    -------
    float
        Scaled SN Absolute Magnitude.

    """
    # mb = 5 * log10(c/H0_jla * Dl(z)) + 25 + MB_jla
    # mb = 5 * log10(c/HO_True * Dl(z)) + 25 + MB_jla - 5 * log10(1 + dH0)
    # with dH0 = (H0_jla - H0_True)/ H0_True
    # MB_True = MB_jla - 5 * log10(1 + dH0)

    # Scale the H0 value of JLA to the H0 value of sim
    h_jla = 0.7  # H0 = 70 Mpc / km / s
    M0_jla = -19.05
    dh = (h_jla - h) / h

    return M0_jla - 5 * np.log10(1 + dh)


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
    if isinstance(date, (int, np.integer, float, np.floating)):
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
    return [z_shell, dist / norm]


def asym_gauss(mu, sig_low, sig_high=None, seed=None, size=1):
    """Generate random parameters using an asymetric Gaussian distribution.

    Parameters
    ----------
    mean : float
        The central value of the Gaussian.
    sig_low : float
        The low sigma.
    sig_high : float
        The high sigma.
    seed : int, optional
        Random seed.
    size: int
        Number of numbers to generate

    Returns
    -------
    numpy.ndarray(float)
        Random(s) variable(s).

    """
    def asym_pdf(x):
        x = np.atleast_1d(x)
        pos = x > mu
        pdf = np.zeros(len(x))

        pdf[pos] = gauss(mu, sig_high, x[pos]) * np.sqrt(2 * np.pi) * sig_high
        pdf[~pos] = gauss(mu, sig_low, x[~pos]) * np.sqrt(2 * np.pi) * sig_low
        norm = np.sqrt(np.pi / 2) * (sig_high + sig_low)
        return pdf / norm

    asym_dist = CustomRandom(asym_pdf, mu - 10 * sig_low, mu + 10 * sig_high,
                             dx=np.min([sig_low, sig_high])/100)
    return asym_dist.draw(size, seed=seed)


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
    coordfk5 = SkyCoord(ra * astu.rad,
                        dec * astu.rad,
                        frame='fk5')  # coord in fk5 frame

    galac_coord = coordfk5.transform_to('galactic')
    l_gal = galac_coord.l.rad - 2 * np.pi * \
        np.sign(galac_coord.l.rad) * (abs(galac_coord.l.rad) > np.pi)
    b_gal = galac_coord.b.rad

    ss = np.sin(b_gal) * np.sin(b_cmb * np.pi / 180)
    ccc = np.cos(b_gal) * np.cos(b_cmb * np.pi / 180) * np.cos(l_gal - l_cmb * np.pi / 180)
    return (1 - v_cmb * (ss + ccc) / C_LIGHT_KMS) - 1.


def init_sn_model(name, model_dir=None):
    """Initialise a sncosmo model.

    Parameters
    ----------
    name : str
        Name of the model.
    model_dir : str
        Path to the model files.

    Returns
    -------
    sncosmo.Model
        sncosmo Model corresponding to input configuration.
    """
    if model_dir is None:
        return snc.Model(source=name)
    else:
        if name == 'salt2':
            return snc.Model(source=snc.SALT2Source(model_dir, name='salt2'))
        elif name == 'salt3':
            return snc.Model(source=snc.SALT3Source(model_dir, name='salt3'))
    return None


def snc_fitter(lc, fit_model, fit_par, **kwargs):
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
        res = snc.fit_lc(data=lc, model=fit_model,
                         vparam_names=fit_par, **kwargs)
        if res[0]['covariance'] is None:
            res[0]['covariance'] = np.empty((len(res[0]['vparam_names']),
                                             len(res[0]['vparam_names'])))
            res[0]['covariance'][:] = np.nan

        res[0]['param_names'] = np.append(res[0]['param_names'], 'mb')
        res[0]['parameters'] = np.append(res[0]['parameters'],
                                         res[1].source_peakmag('bessellb', 'ab'))

        res_dic = {k: v for k, v in zip(res[0]['param_names'], res[0]['parameters'])}
        res = np.append(res, res_dic)
    except (RuntimeError, snc.fitting.DataQualityError):
        res = ['NaN', 'NaN', 'NaN']
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
    nu, dnu = snc.utils.integration_grid(
        snc.constants.C_AA_PER_S / b.maxwave(),
        snc.constants.C_AA_PER_S / b.minwave(),
        snc.constants.C_AA_PER_S / snc.constants.MODEL_BANDFLUX_SPACING)

    trans = b(snc.constants.C_AA_PER_S / nu)
    trans_int = np.sum(trans / nu) * dnu / snc.constants.H_ERG_S
    norm = 10**(-0.4 * zp) * magsys.zpbandflux(b) / trans_int * 10**23 * 10**6
    return norm

def zobs_MinT_MaxT(par, model_t_range):
    zobs = (1. + par['zcos']) * (1. + par['z2cmb']) * (1. + par['vpec'] / C_LIGHT_KMS) - 1.
    MinT = par['sim_t0'] + model_t_range[0] * (1. + zobs)
    MaxT = par['sim_t0'] + model_t_range[1] * (1. + zobs)
    return zobs, MinT, MaxT

def print_dic(dic, prefix=''):
    indent = '    '
    for K in dic:
        if isinstance(dic[K], dict):
            print(prefix + K + ':')
            print_dic(dic[K], prefix=prefix + indent)
        else:
            print(prefix + f'{K}: {dic[K]}')
