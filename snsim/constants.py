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