"""This module contains all the constants used in the package"""
import re
from pathlib import Path
from astropy import constants as cst

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

# just an offset -> set_peakmag(mb=0,'bessellb', 'ab') ->
# offset=2.5*log10(get_x0) change with magsys
SNC_MAG_OFFSET_AB = 10.5020699

VCMB = 620
L_CMB = 271.0
B_CMB = 29.6

SEP = '###############################################'
