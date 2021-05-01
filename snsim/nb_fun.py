from numba import njit
import numpy as np

@njit(cache=True)
def R_base(a,t,vec):
    """Return the new coordinates in field frame a=ra,t=-dec, matrix computed using sagemaths"""
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

@njit(cache=True)
def new_coord_on_fields(ra_frame,dec_frame,vec):
    """Iter on fields to compute the new location of the sn in each new field"""
    new_vec=np.zeros((len(ra_frame),3))
    new_radec=np.zeros((len(ra_frame),2))
    for i in range(len(ra_frame)):
        x,y,z = R_base(ra_frame[i],-dec_frame[i],vec)
        new_radec[i][0] = np.arctan2(y,x)
        new_radec[i][1] = np.arcsin(z)
    return new_radec.T
