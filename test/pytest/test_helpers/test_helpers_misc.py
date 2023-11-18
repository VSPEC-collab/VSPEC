

from astropy import units as u
import numpy as np
import pytest
from pathlib import Path
import pandas as pd

from VSPEC import helpers


DATA_DIR = Path(__file__).parent / '..' / 'data' / 'test_analysis'
EMPTY_DIR = Path(__file__).parent / '..' / 'data' / 'empty'


def test_get_transit_radius():
    """
    Test `get_transit_radius()`

    Run tests for VSPEC.helpers.get_transit_radius()
    This function calculates the minimum radius from mid-transit
    (i.e. phase=180, i=90 deg) that a planet must be to have no
    overlap between the planetary disk and stellar disk.
    """
    # Use parameters for GJ 1214 b
    # Taken from NExSci Archive on
    # 2022-03-03 by Ted Johnson
    system_distance = 14.6427*u.pc
    stellar_radius = 0.215*u.R_sun
    semimajor_axis = 0.01490*u.AU
    planet_radius = 2.742*u.R_earth
    transit_duration = 0.8788*u.hr
    orbital_period = 1.58040433*u.day
    true_radius = np.pi*u.rad * \
        (transit_duration/orbital_period).to_value(u.dimensionless_unscaled)
    predicted_radius = helpers.get_transit_radius(system_distance,
                                                  stellar_radius,
                                                  semimajor_axis,
                                                  planet_radius)
    true_radius = true_radius.to_value(u.deg)
    predicted_radius = predicted_radius.to_value(u.deg)
    message = f'True: {true_radius:.2f}, Calculated: {predicted_radius:.2f}'
    assert predicted_radius == pytest.approx(
        true_radius, rel=0.1), 'test failed: ' + message


def test_get_planet_indices():
    planet_times = np.array([0, 1, 2, 3, 4]) * u.day
    tindex = 2.5 * u.day

    N1, N2 = helpers.get_planet_indicies(planet_times, tindex)

    planet_times = np.array([0, 1, 2, 3, 4]) * u.day
    tindex = 2 * u.day
    N1, N2 = helpers.get_planet_indicies(planet_times, tindex)

    assert N1 == 2
    assert N2 == 2

@pytest.mark.skip()
def test_read_lyr():
    """
    Test `VSPEC.helpers.read_lyr()`
    """
    file = DATA_DIR / 'layer00000.csv'
    fake_file = EMPTY_DIR / 'layer00000.csv'
    wrong_file = DATA_DIR / 'phase00000.csv'

    with pytest.raises(FileNotFoundError):
        helpers.read_lyr(fake_file)
    with pytest.raises(ValueError):
        helpers.read_lyr(wrong_file)
    data = helpers.read_lyr(file)
    assert isinstance(data, pd.DataFrame)
