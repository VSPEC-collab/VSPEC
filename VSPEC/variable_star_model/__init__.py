"""
Variable Star Module
--------------------

This module defines the custom star that ``VSPEC`` uses to 
add stellar variability to its datasets.

Sources of variability include star spots, faculae, flares, and granulation.
"""
from VSPEC.variable_star_model.star import Star
from VSPEC.variable_star_model.spots import StarSpot, SpotCollection, SpotGenerator
from VSPEC.variable_star_model.faculae import Facula, FaculaCollection, FaculaGenerator
from VSPEC.variable_star_model.flares import StellarFlare, FlareCollection, FlareGenerator
