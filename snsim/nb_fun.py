"""This module contains functions with numba decorator to speed up the simulation."""
from numba import njit
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
def R_base(a, t, vec, to_field_frame=True):
    """Return the new carthesian coordinates after z axis and vec axis rotations.

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

    if to_field_frame:
        return R.T @ vec
    else:
        return R @ vec


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
    """Return the index of the first occurence of item in vec."""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


@njit(cache=True)
def time_selec(expMJD, t0, ModelMaxT, ModelMinT, fieldID):
    """Select observations that are made in the good time to see a t0 peak SN.

    Parameters
    ----------
    expMJD : numpy.ndarray(float)
        MJD date of observations.
    t0 : numpy.ndarray(float)
        SN peakmjd.
    ModelMaxT : float
        SN model max date.
    ModelMinT : float
        SN model minus date.
    fiveSigmaDepth :
        5 sigma mag limit of observations.
    fieldID : numpy.ndarray(float)
        Field Id of each observation.

    Returns
    -------
    numpy.ndarray(bool), numpy.ndarray(int)
        The boolean array of field selection and the id of selectionned fields.
    """
    epochs_selec = (expMJD - t0 > ModelMinT) & \
                   (expMJD - t0 < ModelMaxT)
    return epochs_selec, np.unique(fieldID[epochs_selec])


@njit(cache=True)
def map_obs_fields(epochs_selec, fieldID, obsfield):
    """Return the boolean array corresponding to observed fields.

    Parameters
    ----------
    epochs_selec : numpy.array(bool)
        Actual observations selection.
    fieldID : numpy.array(int)
        ID of fields.
    mapdic : numba.Dict(int:bool)
        Numba dic of observed subfield in each observed field.

    Returns
    -------
    bool, numpy.ndarray(bool)
        Is there an observation and the selection of observations.

    """
    epochs_selec[np.copy(epochs_selec)] &= np.array([True if fID in obsfield
                                                    else False for fID in fieldID])
    return epochs_selec.any(), epochs_selec


@njit(cache=True)
def map_obs_subfields(epochs_selec, obs_fieldID, obs_subfield, mapdic):
    """Return boolean array corresponding to observed subfields.

    Parameters
    ----------
    epochs_selec : numpy.array(bool)
        Actual observations selection.
    obs_fieldID : bool
        Id of pre selected observed fields.
    obs_subfield : bool
        Observed subfields.
    mapdic : numba.Dict(int:int)
        Numba dic of observed subfield in each observed field.

    Returns
    -------
    bool, numpy.ndarray(bool)
        Is there an observation and the selection of observations.

    """
    epochs_selec[np.copy(epochs_selec)] &= (obs_subfield == np.array([mapdic[field] for field in
                                                                     obs_fieldID]))
    return epochs_selec.any(), epochs_selec


@njit(cache=True)
def radec_to_cart_2d(ra, dec):
    """Compute carthesian vector for given RA Dec coordinates.

    Parameters
    ----------
    ra : float or numpy.ndarray
        Right Ascension.
    dec :  float or numpy.ndarray
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
    ra : float or numpy.ndarray
        Right Ascension.
    dec :  float or numpy.ndarray
        Declinaison.

    Returns
    -------
    numpy.ndarray(float)
        Carthesian coordinates corresponding to RA Dec coordinates.

    """
    cart_vec = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
    return cart_vec


@njit(cache=True)
def is_in_field(SN_ra, SN_dec, ra_fields, dec_fields, obs_fieldID,
                subfields_id, subfields_corners, type=types.float64[::1]):
    """Chek if a SN is in fields.

    Parameters
    ----------
    epochs_selec : numpy.ndarray(bool)
        The boolean array of field selection.
    obs_fieldID : numpy.ndarray(int)
        Field Id of each observation.
    ra_field_frame : numpy.ndaray(float)
        SN Right Ascension in fields frames.
    dec_field_frame : numpy.ndaray(float)
        SN Declinaison in fields frames.
    field_size : list(float)
        ra and dec size.
    fieldID : numpy.ndarray(int)
        The list of preselected fields ID.

    Returns
    -------
    numba.Dict(int:bool), numba.Dict(int:numpy.array(float))
        The dictionnaries of boolean selection of obs fields and coordinates in observed fields.

    """
    ra_fields_array = np.atleast_1d(ra_fields)
    dec_fields_array = np.atleast_1d(dec_fields)
    vec = radec_to_cart(SN_ra, SN_dec)

    SN_ra_field_frame, SN_dec_field_frame = new_coord_on_fields(ra_fields_array,
                                                                dec_fields_array,
                                                                vec)
    # Check if the SN is in the field
    # in_field = np.abs(SN_ra_field_frame) < field_size[0] / 2
    # in_field &= np.abs(SN_dec_field_frame) < field_size[1] / 2

    dic_map = Dict.empty(key_type=types.i8,
                         value_type=types.i8)

    for i, fID in enumerate(obs_fieldID):
        for subf, subf_id in zip(subfields_corners, subfields_id):
            obs_condition = SN_ra_field_frame[i] > np.min(subf.T[0])
            obs_condition &= SN_ra_field_frame[i] < np.max(subf.T[0])
            obs_condition &= SN_dec_field_frame[i] > np.min(subf.T[1])
            obs_condition &= SN_dec_field_frame[i] < np.max(subf.T[1])
            if obs_condition:
                dic_map[fID] = subf_id
                break
    return dic_map


@njit(cache=True)
def find_idx_nearest_elmt(val, array, treshold):
    """Find the index of the nearest element of array relative to val.

    Parameters
    ----------
    val : float
        A float number.
    array : numpy.ndarray(float)
        An array of float.
    treshold : float
        The maximum gap between val and the nearest element.

    Returns
    -------
    int
        The index of the nearest element.

    """
    smallest_diff_idx = []
    for v in val:
        diff_array = np.abs(array - v)
        idx = diff_array.argmin()
        if diff_array[idx] > treshold:
            raise RuntimeError('Difference above threshold')
        smallest_diff_idx.append(idx)
    return smallest_diff_idx
