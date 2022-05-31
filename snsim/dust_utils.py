"""This module contains dust features."""

import os
import sncosmo as snc
import sfdmap
from snsim import __snsim_dir_path__
import glob
import requests
import tarfile


def check_files_and_download():
    """Check if sdfmap files are here and download if not.

    Returns
    -------
    None
        No return, just download files.
    
    Notes
    -----
    TODO : Change that for environement variable or cleaner solution

    """
    files_in_dust_data = glob.glob(__snsim_dir_path__ + '/dust_data/*.fits')
    files_list = ['SFD_dust_4096_ngp.fits', 'SFD_dust_4096_sgp.fits',
                  'SFD_mask_4096_ngp.fits', 'SFD_mask_4096_sgp.fits']
    filenames = []
    for file in files_in_dust_data:
        filenames.append(os.path.basename(file))
    for file in files_list:
        if file not in filenames:
            print("Dowloading sfdmap files from https://github.com/kbarbary/sfddata/")
            url = "https://github.com/kbarbary/sfddata/archive/master.tar.gz"
            response = requests.get(url, stream=True)
            file = tarfile.open(fileobj=response.raw, mode="r|gz")
            file.extractall(path=__snsim_dir_path__ + '/dust_data')
            new_file = glob.glob(__snsim_dir_path__ + '/dust_data/sfddata-master/*.fits')
            for nfile in new_file:
                os.replace(nfile, __snsim_dir_path__ + '/dust_data/' + os.path.basename(nfile))
            other_files = glob.glob(__snsim_dir_path__ + '/dust_data/sfddata-master/*')
            for ofile in other_files:
                os.remove(ofile)
            os.rmdir(__snsim_dir_path__ + '/dust_data/sfddata-master')
            break


def init_mw_dust(model, mw_dust):
    """Set MW dut effect on sncosmo model.

    Parameters
    ----------
    model : sncosmo.Model
        The sncosmo model which to add the mw dust.
    mw_dust_mod : dic
        The model of dust to apply.

    Returns
    -------
    None
        Directly modify the sncosmo model.

    """
    f99_r_v = 3.1
    if 'rv' in mw_dust:
        f99_r_v = mw_dust['rv']
    if mw_dust['model'].lower() == 'ccm89':
        dust = snc.CCM89Dust()
    elif mw_dust['model'].lower() == 'od94':
        dust = snc.OD94Dust()
    elif mw_dust['model'].lower() == 'f99':
        dust = snc.F99Dust(r_v=f99_r_v)
    else:
        raise ValueError(f"{mw_dust['model']} model does not exist in sncosmo")

    model.add_effect(dust, frame='obs', name='mw_')


def add_mw_to_fit(fit_model, mwebv, mod_name, rv=3.1):
    """Set mw model parameters of a sncsomo model.

    Parameters
    ----------
    fit_model : type
        Description of parameter `fit_model`.
    mwebv : float
        E(B-V) color excess of the sn.
    rv : float
        R_v coeff of the MW.

    Returns
    -------
    None
        Directly modify the sncosmo model.

    """
    if 'mw_' in fit_model.effect_names:
        fit_model.set(mw_ebv=mwebv)
    if mod_name .lower() not in ['f99']:
        fit_model.set(mw_r_v=rv)


def compute_ebv(ra, dec):
    """Compute E(B-V) color excess.

    Parameters
    ----------
    ra : float or numpy.ndarray
        Right Ascension.
    dec : float or numpy.ndarray
        Declinaison.

    Returns
    -------
    float or numpy.ndarray
        The color excess correponding to ra, dec coordinates.

    """
    map = sfdmap.SFDMap(__snsim_dir_path__ + '/dust_data')
    ebv = map.ebv(ra, dec, unit='radian')
    return ebv
