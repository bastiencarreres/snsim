"""This module contains functions with numba decorator to speed up the simulation."""
from numba import njit, prange, guvectorize
import numpy as np
from numba.core import types
from numba.typed import Dict


@njit(cache=True)
def sine_interp(x_new, fun_x, fun_y):
    """Return the sinus interpolation of a function at x.

    Parameters
    ----------
    x_new : float
        New x where evaluate the function.
    fun_x : list(float)
        Existing function x.
    fun_y : list(float)
        Existing function y.

    Returns
    -------
    float
        The sinus interpolation value of f at x_new.

    """
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
        idx_inf = np.where(inf_sel * sup_sel)[0][0]

    x_inf = fun_x[idx_inf]
    x_sup = fun_x[idx_inf + 1]
    Value_inf = fun_y[idx_inf]
    Value_sup = fun_y[idx_inf + 1]
    sin_interp = np.sin(np.pi * (x_new - 0.5 * (x_inf + x_sup)) / (x_sup - x_inf))

    return 0.5 * (Value_sup + Value_inf) + 0.5 * (Value_sup - Value_inf) * sin_interp


@njit(cache=True)
def R_base(theta, phi, vec):
    """Give rotation to RA := theta, DEC := phi.
    Parameters
    ----------
    theta : float
        RA amplitude of the rotation
    phi : float
        Dec amplitude of the rotation 
    vec : numpy.ndarray(float)
        Carthesian vector to rotate
    Returns
    -------
    numpy.ndarray(float)
        Rotated vector.
    """
    R = np.zeros((3, 3), dtype='float')
    R[0, 0] = np.cos(phi) * np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[0, 2] = -np.cos(theta) * np.sin(phi)
    R[1, 0] = np.cos(phi) * np.sin(theta)
    R[1, 1] = np.cos(theta)
    R[1, 2] = -np.sin(phi) * np.sin(theta)
    R[2, 0] = np.sin(phi)
    R[2, 1] = 0
    R[2, 2] = np.cos(phi)
    return R @ vec


@guvectorize(["(float64[:, :, :], float64[:, :], float64[:, :, :])"],
              "(m, n, k),(k, m)->(m, n, k)", nopython=True)
def new_coord_on_fields(ra_dec, ra_dec_frame, new_radec):
    """Compute new coordinates of an object in a list of fields frames.
    Parameters
    ----------
    ra_frame : numpy.ndarray(float)
        Field Right Ascension.
    dec_frame : numpy.ndarray(float)
        Field Declinaison.
    vec : numpy.ndarray(float, size = 3)
        The carthesian coordinates of the object.
    Returns
    -------
    numpy.ndarray(float, size = (2, ?))
        The new coordinates of the obect in each field frame.
    """
    for i in range(ra_dec_frame.shape[1]):
        ra = ra_dec[i]
        vec = np.vstack((np.cos(ra_dec[i, :, 0]) * np.cos(ra_dec[i, :, 1]),
                         np.sin(ra_dec[i, :, 0]) * np.cos(ra_dec[i, :, 1]),
                         np.sin(ra_dec[i, :, 1])))
        x, y, z = R_base(ra_dec_frame[0, i], ra_dec_frame[1, i], vec)
        new_radec[i, :, 0] = np.arctan2(y, x)
        new_radec[i, :, 0][new_radec[i, :, 0] < 0] +=  2 * np.pi
        new_radec[i, :, 1] = np.arcsin(z)


@njit(cache=True)
def radec_to_cart_2d(ra, dec):
    """Compute carthesian vector for given RA Dec coordinates.

    Parameters
    ----------
    ra : numpy.ndarray
        Right Ascension.
    dec : numpy.ndarray
        Declinaison.

    Returns
    -------
    numpy.ndarray(float)
        Carthesian coordinates corresponding to RA Dec coordinates.

    """
    cart_vec = np.vstack((np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec))).T
    return cart_vec


@njit(cache=True)
def radec_to_cart(ra, dec):
    """Compute carthesian vector for given RA Dec coordinates.

    Parameters
    ----------
    ra : float
        Right Ascension.
    dec :  float
        Declinaison.

    Returns
    -------
    numpy.array(float)
        Carthesian coordinates corresponding to RA Dec coordinates.

    """
    cart_vec = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
    return cart_vec