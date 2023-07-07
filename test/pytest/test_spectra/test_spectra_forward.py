
import numpy as np
from astropy import units as u
from VSPEC.spectra import ForwardSpectra


def test_forward_spectra_blackbody():
    wl = np.asarray([1, 2, 3])*u.um
    teff = 5000.0*u.K
    forward_spectra = ForwardSpectra.blackbody()
    flux = forward_spectra.evaluate(wl, teff)

    # Assertions
    assert flux.shape == (3,)
    assert np.all(flux > 0)
