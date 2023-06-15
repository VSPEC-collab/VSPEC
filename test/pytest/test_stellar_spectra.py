#!/usr/bin/env python

from pathlib import Path
import pytest
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt



from VSPEC.stellar_spectra import fast_bin_raw_data, bin_raw_data,get_phoenix_path
from VSPEC.stellar_spectra import bin_from_cache

def test_bin_raw_data():
    """
    Test for `bin_raw_data
    """
    path = get_phoenix_path(3300.0)
    R = 50
    w1 = 1*u.um
    w2 = 18*u.um
    wl,fl = bin_raw_data(path,50,w1,w2)
    assert fl.unit == u.Unit('W m-2 um-1')
    assert wl.unit == u.Unit('um')
    assert wl.shape == fl.shape
    assert np.all(wl >= w1)
    assert np.all(wl <= w2)
    
    R = 50
    print(f'R={R}')
    wl,fl = bin_raw_data(path,R,w1,w2)
    plt.plot(wl,fl)
    R = 200
    print(f'R={R}')
    wl,fl = bin_raw_data(path,R,w1,w2)
    plt.plot(wl,fl)
    R = 500
    print(f'R={R}')
    wl,fl = bin_raw_data(path,R,w1,w2)
    plt.plot(wl,fl)
    # R = 2000
    # print(f'R={R}')
    # wl,fl = bin_raw_data(path,R,w1,w2)
    # R = 10000
    # print(f'R={R}')
    # wl,fl = bin_raw_data(path,R,w1,w2)
    0

def test_bin_cached_data():
    """
    Test for `bin_raw_data
    """
    
    R = 50
    w1 = 1*u.um
    w2 = 18*u.um
    wl,fl = bin_from_cache(3300,R,w1,w2)
    assert fl.unit == u.Unit('W m-2 um-1')
    assert wl.unit == u.Unit('um')
    assert wl.shape == fl.shape
    assert np.all(wl >= w1)
    assert np.all(wl <= w2)
    
    R = 50
    print(f'R={R}')
    wl,fl = bin_from_cache(3300,R,w1,w2)
    plt.plot(wl,fl)
    R = 200
    print(f'R={R}')
    wl,fl = bin_from_cache(3300,R,w1,w2)
    plt.plot(wl,fl)
    R = 500
    print(f'R={R}')
    wl,fl = bin_from_cache(3300,R,w1,w2)
    plt.plot(wl,fl)
    # R = 2000
    # print(f'R={R}')
    # wl,fl = bin_raw_data(path,R,w1,w2)
    # R = 10000
    # print(f'R={R}')
    # wl,fl = bin_raw_data(path,R,w1,w2)
    0


def test_compare_binning():
    path = get_phoenix_path(3300.0)
    R = 50
    w1 = 1*u.um
    w2 = 18*u.um
    wl,fl = bin_raw_data(path,50,w1,w2)
    wl_new, fl_new = bin_from_cache(3300,R,w1,w2)
    plt.plot(wl,1e6*(fl_new-fl)/fl)
    0


if __name__ in '__main__':
    # test_bin_raw_data()
    test_bin_cached_data()
    test_compare_binning()