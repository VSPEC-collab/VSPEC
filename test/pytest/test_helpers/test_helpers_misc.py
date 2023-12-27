

from astropy import units as u
import numpy as np
import pytest
from pathlib import Path
import pandas as pd

from VSPEC import helpers


DATA_DIR = Path(__file__).parent / '..' / 'data' / 'test_analysis'
EMPTY_DIR = Path(__file__).parent / '..' / 'data' / 'empty'

def test_get_planet_indices():
    planet_times = np.array([0, 1, 2, 3, 4]) * u.day
    tindex = 2.5 * u.day

    N1, N2 = helpers.get_planet_indicies(planet_times, tindex)

    planet_times = np.array([0, 1, 2, 3, 4]) * u.day
    tindex = 2 * u.day
    N1, N2 = helpers.get_planet_indicies(planet_times, tindex)

    assert N1 == 2
    assert N2 == 2
