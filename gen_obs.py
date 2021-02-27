import numpy as np
from astropy.table import Table
from astropy.io import fits

def gen_obs(n_obs,n_epochs_b,bands,zp,mjdstart,ra_list,dec_list,magsys='ab',gain=1.000):
    '''Write the obs file in fits format'''
    dtype = [('time',np.dtype('f8')),
             ('band',np.dtype('U25')),
             ('gain',np.dtype('f8')),
             ('skynoise',np.dtype('f8')),
             ('zp',np.dtype('f8')),
             ('zpsys',np.dtype('U25'))]

    bands_str=''
    for b in bands:
        bands_str += b+' '

    obs_hdu_list=[]
    mjd = mjdstart
    skynoise = 1./5*10**(0.4*(mean_depth-20.5))
    for i in range(n_obs):
        ra = ra_list[i]
        dec = dec_list[i]
        obs_array=[]
        for j in range(n_epochs_b):
            mjd+=1.5
            for k,b in enumerate(bands):
                obs_array.append((mjd,b,gain,skynoise,mean_depth,magsys))
        obs_array=np.asarray(obs_array,dtype=dtype)
        tab = Table(data=obs_array, meta={'ra': ra ,'dec': dec, 'n_epochs': n_epochs_b*len(bands)})
        obs_hdu_list.append(fits.table_to_hdu(tab))
    hdu_list = fits.HDUList([fits.PrimaryHDU(header=fits.Header({'n_obs': n_obs, 'bands': bands_str}))]+obs_hdu_list)
    hdu_list.writeto('obs_file.fits',overwrite=True)
