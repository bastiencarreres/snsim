"""This module contains all the constants used in the package."""
import re
from pathlib import Path
from astropy import constants as cst
import numpy as np
import shapely.geometry as shp_geo

path_location = Path(__file__).absolute().parent
init_location = path_location / '__init__.py'
VERSION = re.findall(r"__version__ = \"(.*?)\"",
                     init_location.open().read())[0]
SN_SIM_PRINT = '      _______..__   __.         _______. __  .___  ___. \n'
SN_SIM_PRINT += '     /       ||  \\ |  |        /       ||  | |   \\/   | \n'
SN_SIM_PRINT += '    |   (----`|   \\|  |       |   (----`|  | |  \\  /  | \n'
SN_SIM_PRINT += '     \\   \\    |  . `  |        \\   \\    |  | |  |\\/|  | \n'
SN_SIM_PRINT += ' .----)   |   |  |\\   |    .----)   |   |  | |  |  |  | \n'
SN_SIM_PRINT += ' |_______/    |__| \\__|    |_______/    |__| |__|  |__| \n'
SN_SIM_PRINT += f'================================= Version : {VERSION} ====== '

# Light velocity in km/s
C_LIGHT_KMS = cst.c.to('km/s').value

# CMB DIPOLE from Planck18 https://arxiv.org/pdf/1807.06205.pdf
VCMB = 369.82  # km/s
L_CMB = 264.021  # deg
B_CMB = 48.253  # deg

SEP = '###############################################'


_SPHERE_LIMIT_ = shp_geo.LineString([[2 * np.pi, -np.pi/2],
                                     [2 * np.pi,  np.pi/2]])


# M0 SNIA from JLA paper (https://arxiv.org/abs/1401.4064)
SNIA_M0 = {'jla': -19.05}


# SNII mean and scattering of luminosity function values from Vincenzi et al. 2021 Table 5 (https://arxiv.org/abs/2111.10382)
SNCC_M0 = {
           'SNIIpl': {'li11_gaussian': -15.97, 'li11_skewed': -17.51},
           'SNIIb': {'li11_gaussian': -16.69, 'li11_skewed': -18.30},
           'SNIIn': {'li11_gaussian': -17.90, 'li11_skewed': -19.13},
           'SNIc': {'li11_gaussian': -16.75, 'li11_skewed': -17.51},
           'SNIb': {'li11_gaussian': -16.07, 'li11_skewed': -17.71},
           'SNIc_BL': {'li11_gaussian': -16.79, 'li11_skewed': -17.74}
          }

SNCC_mgscatter = { 
                  'SNIIpl': {'li11_gaussian': [1.31, 1.31], 'li11_skewed': [2.01, 3.18]},
                  'SNIIb': {'li11_gaussian': [1.38, 1.38], 'li11_skewed': [2.03, 7.40]},
                  'SNIIn': {'li11_gaussian': [0.95, 0.95], 'li11_skewed' :[1.53, 6.83]},
                  'SNIc': {'li11_gaussian': [0.97, 0.97], 'li11_skewed': [1.24, 1.22]},
                  'SNIb': {'li11_gaussian': [1.34, 1.34], 'li11_skewed': [2.11, 7.15]},
                  'SNIc_BL': {'li11_gaussian': [0.95, 0.95], 'li11_skewed': [1.35, 2.06]}
                 }

# ztf20 relative fraction of SNe subtypes from https://arxiv.org/abs/2009.01242 figure 6 +
#relative fraction between SNe Ic and SNe Ib from https://iopscience.iop.org/article/10.3847/1538-4357/aa5eb7/meta
#shiver17 fraction from https://arxiv.org/abs/1609.02922
SNCC_fraction = {
                 'shivers17': {
                           'SNIIpl': 0.620136,
                           'SNIIb': 0.10944,
                           'SNIIn': 0.046632,
                           'SNIc': 0.075088,
                           'SNIb': 0.108224,
                           'SNIc_BL': 0.011248
                           },
                'ztf20': {
                           'SNIIpl': 0.546554,
                           'SNIIb': 0.047652,
                           'SNIIn': 0.102524,
                           'SNIc': 0.110357,
                           'SNIb': 0.052551,
                           'SNIc_BL': 0.05421
                            }
                }

#Value of h used in the various articles
h_article = {
             'jla' : 0.70,
             'li11': 0.73
             }
