"""VSPEC variable star module

This module describes the stellar variability
contianed in `VSPEC`'s model.
"""
import astropy.units as u
from numpy import pi
from VSPEC.variable_star_model.star import Star
from VSPEC.variable_star_model.spots import StarSpot, SpotCollection, SpotGenerator
from VSPEC.variable_star_model.faculae import Facula, FaculaCollection, FaculaGenerator
from VSPEC.variable_star_model.flares import StellarFlare, FlareCollection, FlareGenerator


MSH = u.def_unit('micro solar hemisphere', 1e-6 * 0.5 * 4*pi*u.R_sun**2)
"""Micro-solar hemisphere

This is a standard unit in heliophysics that
equals one millionth of one half the surface area of the Sun.
"""