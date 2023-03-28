"""VSPEC variable star module

This module describes the stellar variability
contianed in `VSPEC`'s model.
"""
import astropy.units as u
from numpy import pi

MSH = u.def_unit('micro solar hemisphere', 1e-6 * 0.5 * 4*pi*u.R_sun**2)
"""Micro-solar hemisphere

This is a standard unit in heliophysics that
equals one millionth of one half the surface area of the Sun.
"""