"""This module contains function with numba decorator to speed up the simulation
"""
from numba import njit, prange, jit
import numpy as np
import snsim.utils as ut

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
def R_base(a, t, vec):
    """Return the new carthesian coordinates after z axis and
    vec axis rotations.

    Parameters
    ----------
    a : float
        Rotation angle around z axis.
    t : Rotation angle around vec axis.
    vec : numpy.ndarray(float)
        Coordinates of the second rotation axis.

    Returns
    -------
    numpy.ndarray(float)
        Carthesian coordinates in the new basis.

    Notes
    -----
    Rotation matrix computed using sagemaths

    """
    R = np.zeros((3, 3))
    R[0, 0] = (np.cos(t) - 1) * np.cos(a) * np.sin(a)**2 - \
        ((np.cos(t) - 1) * np.sin(a)**2 - np.cos(t)) * np.cos(a)
    R[0, 1] = (np.cos(t) - 1) * np.cos(a)**2 * np.sin(a) + \
        ((np.cos(t) - 1) * np.sin(a)**2 - np.cos(t)) * np.sin(a)
    R[0, 2] = np.cos(a) * np.sin(t)
    R[1, 0] = (np.cos(t) - 1) * np.cos(a)**2 * np.sin(a) - \
        ((np.cos(t) - 1) * np.cos(a)**2 - np.cos(t)) * np.sin(a)
    R[1, 1] = -(np.cos(t) - 1) * np.cos(a) * np.sin(a)**2 - \
        ((np.cos(t) - 1) * np.cos(a)**2 - np.cos(t)) * np.cos(a)
    R[1, 2] = np.sin(a) * np.sin(t)
    R[2, 0] = -np.cos(a)**2 * np.sin(t) - np.sin(a)**2 * np.sin(t)
    R[2, 1] = 0
    R[2, 2] = np.cos(t)

    return R.T @ vec


@njit(cache=True)
def new_coord_on_fields(ra_frame, dec_frame, vec):
    """Compute new coordinates of an object in a list of fields frames.

    Parameters
    ----------
    ra_frame : numpy.ndarray(float)
        Field Right Ascension.
    dec_frame : numpy.ndarray(float)
        Field Declinaison.
    vec : numpy.ndaray(float, size = 3)
        The carthesian coordinates of the object.

    Returns
    -------
    numpy.ndarray(float, size = (2, ?))
    The new coordinates of the obect in each field frame.

    """
    new_radec = np.zeros((2, len(ra_frame)))
    for i in range(len(ra_frame)):
        x, y, z = R_base(ra_frame[i], -dec_frame[i], vec)
        new_radec[0][i] = np.arctan2(y, x)
        new_radec[1][i] = np.arcsin(z)
    return new_radec


@njit(cache=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1

@njit(cache=True)
def is_in_field(field_id, ra_f_frame, dec_f_frame, f_size, pre_select_fields):
    is_in_field = np.abs(ra_f_frame) < f_size[0] / 2
    is_in_field *= np.abs(dec_f_frame) < f_size[1] / 2

    dic_map={}
    for pf, b in zip(pre_select_fields, is_in_field):
        dic_map[pf] = b

    return [dic_map[id] for id in field_id]
