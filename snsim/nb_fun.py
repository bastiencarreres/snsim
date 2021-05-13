"""This module contains function with numba decorator to speed up the simulation
"""
from numba import njit, prange
import numpy as np

@njit(cache=True)
def R_base(a,t,vec):
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
    R=np.zeros((3,3))
    R[0,0] = (np.cos(t)-1)*np.cos(a)*np.sin(a)**2-((np.cos(t)-1)*np.sin(a)**2-np.cos(t))*np.cos(a)
    R[0,1] = (np.cos(t)-1)*np.cos(a)**2*np.sin(a)+((np.cos(t)-1)*np.sin(a)**2-np.cos(t))*np.sin(a)
    R[0,2] = np.cos(a)*np.sin(t)
    R[1,0] = (np.cos(t)-1)*np.cos(a)**2*np.sin(a)-((np.cos(t)-1)*np.cos(a)**2-np.cos(t))*np.sin(a)
    R[1,1] = -(np.cos(t)-1)*np.cos(a)*np.sin(a)**2-((np.cos(t)-1)*np.cos(a)**2-np.cos(t))*np.cos(a)
    R[1,2] = np.sin(a)*np.sin(t)
    R[2,0] = -np.cos(a)**2*np.sin(t)-np.sin(a)**2*np.sin(t)
    R[2,1] = 0
    R[2,2] = np.cos(t)
    return R.T @ vec

@njit(cache=True, parallel=True)
def new_coord_on_fields(ra_frame,dec_frame,vec):
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
    new_radec = np.zeros((2,len(ra_frame)))
    for i in prange(len(ra_frame)):
        x,y,z = R_base(ra_frame[i],-dec_frame[i],vec)
        new_radec[0][i] = np.arctan2(y,x)
        new_radec[1][i] = np.arcsin(z)
    return new_radec
