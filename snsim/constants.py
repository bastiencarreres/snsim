"""This module contains all the constants used in the package."""

import re
from pathlib import Path
from astropy import constants as cst
import numpy as np
import shapely.geometry as shp_geo
from . import dust_utils as dst_ut
from . import scatter as sct

###########
# GENERAL #
###########
path_location = Path(__file__).absolute().parent
init_location = path_location / "__init__.py"
VERSION = re.findall(r"__version__ = \"(.*?)\"", init_location.open().read())[0]

SN_SIM_PRINT = "      _______..__   __.         _______. __  .___  ___. \n"
SN_SIM_PRINT += "     /       ||  \\ |  |        /       ||  | |   \\/   | \n"
SN_SIM_PRINT += "    |   (----`|   \\|  |       |   (----`|  | |  \\  /  | \n"
SN_SIM_PRINT += "     \\   \\    |  . `  |        \\   \\    |  | |  |\\/|  | \n"
SN_SIM_PRINT += " .----)   |   |  |\\   |    .----)   |   |  | |  |  |  | \n"
SN_SIM_PRINT += " |_______/    |__| \\__|    |_______/    |__| |__|  |__| \n"
SN_SIM_PRINT += f"================================= Version : {VERSION} ====== "

SEP = "###############################################"

#############################################
# PHYSICAL & COSMO CONSTANTS/DEFAULT VALUES #
#############################################

# Light velocity in km/s
C_LIGHT_KMS = cst.c.to("km/s").value

# CMB DIPOLE from Planck18 https://arxiv.org/pdf/1807.06205.pdf
VCMB = 369.82  # km/s
L_CMB = 264.021  # deg
B_CMB = 48.253  # deg



_SPHERE_LIMIT_ = shp_geo.LineString([[2 * np.pi, -np.pi / 2], [2 * np.pi, np.pi / 2]])

########################################
# VALUES FROM PAPER USED IN SIMULATION #
########################################

# Values of h used in the various articles
h_registry = {
    "ztf20": 0.70,  # Perley et al. 2020, DOI: https://doi.org/10.3847/1538-4357/abbd98
    "ptf19": 0.70,  # Frohmaier et al. 2019, DOI: https://doi.org/10.1093/mnras/stz807
    "jla": 0.70, # Betoule et al. 2014, DOI: https://doi.org/10.1051/0004-6361/201423413 
    "li11": 0.73, 
    "sullivan06": 0.70 # Sullivan et al. 2006, DOI: https://doi.org/10.1086/506137
    }

# Values of absolute mag used in the various articles
Mabs_registry = {
    "SNIa": {
        "jla": [-19.05, "bessellb"],  # Betoule et al. 2014, DOI: https://doi.org/10.1051/0004-6361/201423413
    },
    "SNII": {
        "li11@gaussian": [-15.97, "bessellr"],
        "li11@skewed": [-17.51, "bessellr"]
    },
    "SNIIb":{
        "li11@gaussian": [-16.69, "bessellr"], 
        "li11@skewed": [-18.30, "bessellr"]
    },
    "SNIc": {
        "li11@gaussian": [-16.75, "bessellr"], 
        "li11@skewed": [-17.51, "bessellr"]
    },
    "SNIb": {
        "li11@gaussian": [-16.07, "bessellr"], 
        "li11@skewed": [-17.71, "bessellr"]
    },
    "SNIc_BL": {
        "li11@gaussian": [-16.79, "bessellr"],
        "li11@skewed": [-17.74, "besseellr"]
    },
    "SNIax": {
        "plasticc": [0.345, "bessellv"]
    },
    "SNIa91bg":
        {
            "plasticc": [0, "bessellv"]
        }
}

sigMabs_registery = {
    "SNIIb": {
        "li11@gaussian": 1.38, 
        "li11@skewed": [2.03, 7.40]
    },
    "SNIIn": {
        "li11@gaussian": 0.95,
        "li11@skewed": [1.53, 6.83]
    },
    "SNIc": {
        "li11@gaussian": 0.97, 
        "li11@skewed": [1.24, 1.22]
    },
    "SNIb": {
        "li11@gaussian": 1.34, 
        "li11@skewed": [2.11, 7.15]
        },
    "SNIc_BL": {
        "li11@gaussian": 0.95,
        "li11@skewed": [1.35, 2.06]
    },

}

rates_registry = {
    "SNIa" : {
        "ztf20": "lambda z:  2.35e-5",  # Perley et al. 2020, DOI: https://doi.org/10.3847/1538-4357/abbd98
        "ptf19": "lambda z:  2.43e-5",  # Frohmaier et al. 2019, DOI: https://doi.org/10.1093/mnras/stz807
        "ptf19@pw": "lambda z:  2.35e-5 * (1 + z)**1.7",  # Frohmaier et al. 2019, DOI: https://doi.org/10.1093/mnras/stz807
    },
    "SNII": {
         # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * 0.69673",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * 0.776208",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z: 9.10e-5 * 0.69673 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6"
    },
    
    "SNIIpl": {
         # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * 0.620136",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * 0.546554",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z: 9.10e-5 * 0.620136 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6"
    },
    "SNIIb": {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * 0.10944",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * 0.047652",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z: 9.10e-5 * 0.10944 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6"
    },
    "SNIIn": {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * 0.046632",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * 0.102524",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z: 9.10e-5 * 0.046632 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6"
    },
    "SNIb/c": {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * 0.19456",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * 0.21711",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z: 9.10e-5 * 0.19456 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6"
    },
    "SNIc": {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * 0.075088",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * 0.110357",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z: 9.10e-5 * 0.075088 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6"
    },
    "SNIb": {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * 0.108224",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * 0.052551",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z: 9.10e-5 * 0.108224 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6"
    },
    "SNIc_BL": {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z: 1.01e-4 * 0.011248",
        # Rate from  https://arxiv.org/abs/2010.15270
        "ztf20": f"lambda z: 9.10e-5 * 0.05421",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z: 9.10e-5 * 0.011248 * ((1 + z)**2.7/(1 + ((1 + z) / 2.9))**5.6"
    },
    "SNIax": {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z:  2.43e-5 * 0.24",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z:  2.43e-5 * 0.24 * ((1 + z)**2.7 / (1 + ((1 + z) / 2.9))**5.6)",
    },
    "SNIa91bg": {
        # Rate from https://arxiv.org/abs/2009.01242, rates of subtype from figure 6
        "ptf19": f"lambda z:  2.43e-5 * 0.12",
        # Rate from https://arxiv.org/abs/2010.15270, pw from https://arxiv.org/pdf/1403.0007.pdf
        "ptf19@pw": f"lambda z:  2.43e-5 * 0.12 * ((1 + z)**2.7 / (1 + ((1 + z) / 2.9))**5.6)",
    }
}


# value of fitted parameter of SNIa-Host_galaxy from Sullivan et al 2006 https://iopscience.iop.org/article/10.1086/506137/pdf


sullivan_para = {"mass": 5.3 * 1.0e-14, "SFR": 3.9 * 1.0e-4}

