"""The snsim module is design to simulate supernovae lightcurve
in a survey defined by an observations data base.
Moreover the simulation can use a host file to simulate a velocity field.

The package use sncosmo.

Github repository : https://github.com/bcarreres/snsim
"""

from .snsim import *
from .utils import *
from .scatter import *
from .sim_class import *
from .nb_fun import *
from .constants import *

__version__ = "0.2.1"
