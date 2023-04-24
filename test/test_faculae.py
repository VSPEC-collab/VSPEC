"""
Tests for VSPEC.variable_star_model.faculae
"""

from astropy import units as u
import numpy as np
import pytest

from VSPEC.variable_star_model import Facula, FaculaCollection, FaculaGenerator
from VSPEC.helpers import MSH, CoordinateGrid, to_float

def init_facula(**kwargs):
    """
    Make a default Facula for testing.
    """
    return Facula(
        lat = kwargs.get('lat',0*u.deg),
        lon = kwargs.get('lon',0*u.deg),
        Rmax = kwargs.get('Rmax',300*u.km),
        R0 = kwargs.get('R0',100*u.km),
        Teff_floor = kwargs.get('Teff_floor',2900*u.K),
        Teff_wall = kwargs.get('Teff_wall',3500*u.K),
        lifetime = kwargs.get('lifetime',10*u.hr),
        growing = kwargs.get('growing',True),
        floor_threshold = kwargs.get('floor_threshold',20*u.km),
        Zw = kwargs.get('Zw',100*u.km),
        Nlat = kwargs.get('Nlat', 300),
        Nlon = kwargs.get('Nlon', 600),
        gridmaker = kwargs.get('gridmaker', None)
    )

def test_facula_init():
    """
    Test `Facula.__init__()`
    """
    fac = init_facula()
    assert np.all(fac.r > 0*u.deg)
    assert np.all(fac.r < 180*u.deg)
    assert np.max(to_float(fac.r, u.deg)) == pytest.approx(180, abs=1)


