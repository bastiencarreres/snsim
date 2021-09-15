"""Init file of snsim package.

The snsim module is design to simulate supernovae lightcurve in a
survey defined by an observations data base.
Moreover the simulation can use a host file to simulate a velocity field.

The package use sncosmo.

Github repository : https://github.com/bcarreres/snsim
"""
import os
__snsim_dir_path__ = os.path.dirname(__file__)

__version__ = "0.3.6_dev"


from .simu import Simulator
from .open_sim import OpenSim
