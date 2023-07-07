import pytest
from pathlib import Path
import numpy as np
from astropy import units as u
from VSPEC.spectra.phoenix import (
    get_binned_options,
    RawReader,
    BinnedReader,
    read_phoenix,
    write_binned_spectrum,
)


@pytest.fixture
def mock_raw_phoenix_path(monkeypatch, tmp_path):
    """
    Mock the raw PHOENIX path for testing.
    """
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    monkeypatch.setattr("VSPEC.spectra.phoenix.RAW_PHOENIX_PATH", raw_path)
    return raw_path


@pytest.fixture
def mock_binned_phoenix_path(monkeypatch, tmp_path):
    """
    Mock the binned PHOENIX path for testing.
    """
    binned_path = tmp_path / "binned"
    binned_path.mkdir()
    monkeypatch.setattr(
        "VSPEC.spectra.phoenix.BINNED_PHOENIX_PATH", binned_path)
    return binned_path


def test_get_binned_options(mock_binned_phoenix_path, monkeypatch):
    """
    Test for `get_binned_options()` function.
    """
    # Create dummy directories for resolving powers
    powers = [1000, 2000, 5000]
    for power in powers:
        (mock_binned_phoenix_path / f"R_{power:0>6}").mkdir()

    # Mock listdir to return the directory names
    def mock_listdir(path):
        return [str(dir.name) for dir in (mock_binned_phoenix_path).iterdir()]

    monkeypatch.setattr("os.listdir", mock_listdir)

    expected_options = np.array(powers)
    assert np.array_equal(get_binned_options(), expected_options)


def test_raw_reader():
    """
    Test for `RawReader` class.
    """
    reader = RawReader()
    teff = 3000 * u.K
    expected_filename = 'lte03000-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'
    filename = reader.get_filename(teff)
    assert filename == expected_filename

    wl, fl = reader.read(teff)

    assert isinstance(wl, u.Quantity)
    assert isinstance(fl, u.Quantity)
    assert wl.unit == u.um
    assert fl.unit == u.Unit("W m-2 um-1")


def test_binned_reader():
    """
    Test for `BinnedReader` class.
    """
    reader = BinnedReader()
    R = 1000
    teff = 3000 * u.K
    expected_dirname = 'R_001000'
    dirname = reader.get_dirname(R)
    assert dirname == expected_dirname
    expected_filename = 'binned3000StellarModel.txt'
    filename = reader.get_filename(teff)
    assert filename == expected_filename

    wl, fl = reader.read(R, teff)

    assert isinstance(wl, u.Quantity)
    assert isinstance(fl, u.Quantity)
    assert wl.unit == u.um
    assert fl.unit == u.Unit("W m-2 um-1")


@pytest.mark.parametrize(
    'teff, R, w1, w2',
    [
        (3000*u.K, 500, 1*u.um, 3*u.um),
        (3000*u.K, 50, 4*u.um, 10*u.um),
        (3000*u.K, 1000, 1*u.um, 3*u.um),
        (3000*u.K, 2000, 2*u.um, 2.3*u.um),
    ]
)
@pytest.mark.filterwarnings("error")
def test_read_phoenix(teff, R, w1, w2):
    wl_new, fl_new = read_phoenix(teff, R, w1, w2)
    assert isinstance(wl_new, u.Quantity)
    assert isinstance(fl_new, u.Quantity)
    assert wl_new.unit == u.um
    assert fl_new.unit == u.Unit("W m-2 um-1")


def test_write_binned_spectrum(tmp_path):
    """
    Test for `write_binned_spectrum()` function.
    """
    path = tmp_path / "output.txt"
    wavelength = np.arange(1000, 2000) * u.Angstrom
    flux_unit = u.Unit("erg cm-2 s-1 cm-1")
    flux = np.ones_like(wavelength.value) * flux_unit

    write_binned_spectrum(path, wavelength, flux)

    assert path.exists()
    with open(path, "r") as file:
        content = file.read()
        assert f"wavelength[Angstrom], flux[{flux_unit.to_string()}]" in content
        assert all([f"{wavelength[i].value:.6e}, {flux[i].value:.6e}" in content for i in range(
            len(wavelength))])
