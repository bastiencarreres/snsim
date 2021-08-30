"""Contains function related to SALT model."""

import sncosmo as snc
import numpy as np
from .constants import SNC_MAG_OFFSET_AB


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
    return 10**(-0.4 * (mB - SNC_MAG_OFFSET_AB))


def n21_x1_model(z, rand_gen=None):
    """From  Nicolas et al. 2021."""

    z = np.atleast_1d(z)

    # Constants defines in the paper
    a = 0.51
    mu1 = 0.37
    mu2 = -1.22
    sig1 = 0.61
    sig2 = 0.56

    rand1, rand2, rand3 = rand_gen.normal(loc=[mu1, mu1, mu2],
                                          scale=[sig1, sig1, sig2],
                                          size=(len(z), 3)).T

    delta_z = 1 / (1 / 0.87 * 1 / (1 + z)**2.8 + 1)
    X1 = delta_z * rand1
    X1 += (1 - delta_z) * (a * rand2 + (1 - a) * rand3)
    return X1


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


def compute_salt_fit_error(fit_model, cov, band, time_th, zp, magsys='ab'):
    r"""Compute fit error on flux from sncosmo fit covariance x0,x1,c.

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
    According to Fnorm = x0/(1+z)
                        * int_\lambda (M0(\lambda_s, p) + x1 * M1(\lambda_s, p))
                        * 10**(-0.4 * c * CL(\lambda_s))
                        * T_b(\lambda) * \lambda/hc d\lambda * norm_factor
    where norm_factor = 10**(0.4 * ZP_norm)/ZP_magsys. We found :
    dF/dx0 = F/x0

    dF/dx1 = x0/(1+z) * int_\lambda M1(\lambda_s,p)) * 10**(-0.4 * c * CL(\lambda_s))
                                 * T_b(\lambda) * \lambda/hc dlambda * norm_factor

    dF/dc  =  -0.4*ln(10)*x0/(1+z) * int_\lambda (M0(\lambda_s, p) + x1 * M1(\lambda_s, p))
                                   * CL(\lambda_s) * 10**(-0.4 * c *CL(\lambda_s))
                                   * T_b(\lambda) * \lambda/hc dl\ambda * norm_factor

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
