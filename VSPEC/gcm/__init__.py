"""
This is the VSPEC default GCM Module
"""

from .heat_transfer import to_pygcm as vspec_to_pygcm
from .twoface import gen_planet as twoface_to_pygcm
