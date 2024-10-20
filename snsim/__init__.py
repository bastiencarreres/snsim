"""Init file of snsim package.

The snsim module is design to simulate supernovae lightcurve in a
survey defined by an observations data base.
Moreover the simulation can use a host file to simulate a velocity field.

The package use sncosmo.

Github repository : https://github.com/bcarreres/snsim
"""

import os

__snsim_dir_path__ = os.path.dirname(__file__)

__version__ = "0.6.0"

from .simu import Simulator
from .sample import SimSample
from . import post_sim_tools
from . import astrobj
from . import generators
from .survey import SurveyObs
from .hosts import SnHosts
from . import utils
from . import io_utils
from . import scatter
from .plasticc_model import *
