"""
The ``VSPEC`` parameters module.

This classes instruct the behavior of the rest
of the ``VSPEC`` package.
"""
from VSPEC.params.base import BaseParameters
from VSPEC.params.read import InternalParameters
from VSPEC.params.read import Header
from VSPEC.params.planet import PlanetParameters
from VSPEC.params.planet import GravityParameters
from VSPEC.params.planet import SystemParameters
from VSPEC.params.gcm import psgParameters
from VSPEC.params.gcm import gcmParameters
from VSPEC.params.gcm import binaryGCM
from VSPEC.params.gcm import waccmGCM
from VSPEC.params.gcm import vspecGCM
from VSPEC.params.observation import ObservationParameters
from VSPEC.params.observation import InstrumentParameters
from VSPEC.params.observation import TelescopeParameters
from VSPEC.params.observation import CoronagraphParameters
from VSPEC.params.observation import SingleDishParameters
from VSPEC.params.observation import DetectorParameters
from VSPEC.params.observation import ccdParameters
from VSPEC.params.observation import BandpassParameters
from VSPEC.params.stellar import StarParameters
from VSPEC.params.stellar import GranulationParameters
from VSPEC.params.stellar import FlareParameters
from VSPEC.params.stellar import FaculaParameters
from VSPEC.params.stellar import SpotParameters
from VSPEC.params.stellar import LimbDarkeningParameters