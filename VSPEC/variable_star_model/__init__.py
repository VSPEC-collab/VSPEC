"""VSPEC variable star module

This module describes the stellar variability
contianed in `VSPEC`'s model.
"""
from VSPEC.variable_star_model.star import Star
from VSPEC.variable_star_model.spots import StarSpot, SpotCollection, SpotGenerator
from VSPEC.variable_star_model.faculae import Facula, FaculaCollection, FaculaGenerator
from VSPEC.variable_star_model.flares import StellarFlare, FlareCollection, FlareGenerator
