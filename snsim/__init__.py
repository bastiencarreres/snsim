"""Init file of snsim package.

The snsim module is design to simulate supernovae lightcurve in a
survey defined by an observations data base.
Moreover the simulation can use a host file to simulate a velocity field.

The package use sncosmo.

Github repository : https://github.com/bcarreres/snsim
"""

import os
__snsim_dir_path__ = os.path.dirname(__file__)

__version__ = "0.3.13_dev"

from .simu import Simulator
from .sn_sample import SimSample
from . import post_sim_tools
from . import transients
from . import generators
from . import survey_host
from . import utils
from . import io_utils
