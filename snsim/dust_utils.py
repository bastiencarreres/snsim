import sncosmo as snc
import sfdmap
from snsim import __snsim_dir_path__
import numpy as np

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
