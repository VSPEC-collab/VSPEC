"""
End-to-end test 1

The most basic test.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
import libpypsg
import pytest

import VSPEC

libpypsg.docker.set_url_and_run()

CFG_PATH = Path(__file__).parent / 'test1.yaml'
FIG_PATH = Path(__file__).parent / 'out.png'
u_flux = u.Unit('W m-2 um-1')
u_time = u.day
u_wl = u.um


def make_fig(_data: VSPEC.PhaseAnalyzer):
    """
    Make a figure.
    """
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    therm = fig.add_subplot(gs[0, 0])
    im = therm.pcolormesh(
        _data.time.to_value(u_time),
        _data.wavelength.to_value(u_wl),
        _data.thermal.to_value(u_flux)
    )
    fig.colorbar(im, ax=therm, label=f'Flux {_data.thermal.unit}')
    therm.set_xlabel(f'Time ({u_time})')
    therm.set_ylabel(f'Wavelength ({u_wl})')

    spec = fig.add_subplot(gs[0, 1])
    images = 0
    spec.plot(_data.wavelength, 1e6*_data.spectrum('thermal', images, False) /
              _data.spectrum('total', images, False), c='xkcd:azure', label='True')
    spec.errorbar(_data.wavelength, 1e6*_data.spectrum('thermal', images, True)/_data.spectrum('total', images, False), c='xkcd:rose pink', label='Observed',
                  yerr=1e6*_data.spectrum('noise', images, False)/_data.spectrum('total', images, False), fmt='o', markersize=6)
    spec.set_ylabel('Flux (ppm)')
    spec.set_xlabel(f'Wavelength ({_data.wavelength.unit})')
    spec.legend()

    lc = fig.add_subplot(gs[1, :])
    for i in [0, 20, 40, 80, 145]:
        lc.plot(_data.time, _data.lightcurve('star', i, normalize=0),
                label=f'{_data.wavelength[i]:.1f}')
    lc.set_ylabel('Flux (normalized)')
    lc.set_xlabel(f'Time ({_data.time.unit})')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.savefig(FIG_PATH, facecolor='w')


def read_data() -> VSPEC.PhaseAnalyzer:
    """
    Read the data from disk.
    """
    model = VSPEC.ObservationModel.from_yaml(CFG_PATH)
    return VSPEC.PhaseAnalyzer(model.directories['all_model'])


@pytest.fixture
def _thermal():
    """
    The saved thermal data.
    """
    path = Path(__file__).parent / 'thermal.npy'
    dat = np.load(path)
    return dat*u_flux


@pytest.fixture
def _total():
    """
    The saved total data.
    """
    path = Path(__file__).parent / 'total.npy'
    dat = np.load(path)
    return dat*u_flux


@pytest.fixture
def _noise():
    """
    The saved noise data.
    """
    path = Path(__file__).parent / 'noise.npy'
    dat = np.load(path)
    return dat*u_flux


def test_output(test1_data: VSPEC.PhaseAnalyzer, _thermal, _total, _noise):
    """
    Compare this run to the saved data.
    """
    assert np.all(np.isclose(test1_data.thermal.to_value(u_flux),
                  _thermal.to_value(u_flux),rtol=1e-2)), f'Thermal different by {np.max((test1_data.thermal - _thermal)/_thermal)*100:.1f}%'
    assert np.all(np.isclose(test1_data.total.to_value(u_flux),
                  _total.to_value(u_flux),rtol=1e-2)), f'Total different by {np.max((test1_data.total - _total)/_total*100):.1f}%'
    assert np.all(np.isclose(test1_data.noise.to_value(u_flux), _noise.to_value(u_flux), rtol=1e-2
                  )), f'Noise different by {np.max((test1_data.noise - _noise)/_noise*100):.1f}%'

def save_data(data: VSPEC.PhaseAnalyzer):
    """
    Save the data to disk.
    """
    np.save(Path(__file__).parent / 'thermal.npy', data.thermal.to_value(u_flux))
    np.save(Path(__file__).parent / 'total.npy', data.total.to_value(u_flux))
    np.save(Path(__file__).parent / 'noise.npy', data.noise.to_value(u_flux))


def run():
    """
    Run the model.
    """
    model = VSPEC.ObservationModel.from_yaml(CFG_PATH)
    model.build_planet()
    model.build_spectra()

    _data = VSPEC.PhaseAnalyzer(model.directories['all_model'])
    make_fig(_data)


if __name__ in '__main__':
    pytest.main(args=[Path(__file__), '--test1'])
    data = read_data()
    make_fig(data)
