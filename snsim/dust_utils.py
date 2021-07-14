import os
import sncosmo as snc
import sfdmap
from snsim import __snsim_dir_path__
import numpy as np
import glob
import requests
import tarfile
import sys

def check_files_and_dowload():
    files_in_dust_data = glob.glob(__snsim_dir_path__+'/dust_data/*.fits')
    files_list = ['SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits', 'SFD_mask_4096_ngp.fits', 'SFD_mask_4096_sgp.fits']
    filenames = []
    for file in files_in_dust_data:
        filenames.append(os.path.basename(file))
    for file in files_list:
        if file not in filenames:
            print("Dowloading sfdmap files from https://github.com/kbarbary/sfddata/")
            url = "https://github.com/kbarbary/sfddata/archive/master.tar.gz"
            response = requests.get(url, stream=True)
            file = tarfile.open(fileobj=response.raw, mode="r|gz")
            file.extractall(path= __snsim_dir_path__+'/dust_data')
            new_file = glob.glob(__snsim_dir_path__+'/dust_data/sfddata-master/*.fits')
            for nfile in new_file:
                os.replace(nfile,__snsim_dir_path__+'/dust_data/'+os.path.basename(nfile))
            other_files= glob.glob(__snsim_dir_path__+'/dust_data/sfddata-master/*')
            for ofile in other_files:
                os.remove(ofile)
            os.rmdir(__snsim_dir_path__+'/dust_data/sfddata-master')
            break

def init_mw_dust(model, mw_dust_mod):
    if isinstance(mw_dust_mod, (list, np.ndarray)):
        mw_dust_mod = mw_dust_mod[0]
    if mw_dust_mod.lower() == 'ccm89':
        dust = snc.CCM89Dust()
    elif mw_dust_mod.lower() == 'od94':
        dust = snc.OD94Dust()
    elif mw_dust_mod.lower == 'F99' :
        dust = snc.F99Dust()
    else:
        raise ValueError(f'{mw_dust_mod} model does not exist in sncosmo')

    model.add_effect(dust, frame='obs', name='mw_')

def compute_ebv(ra, dec):
    map = sfdmap.SFDMap(__snsim_dir_path__+'/dust_data')
    ebv = map.ebv(ra, dec)
    return ebv
