
from pathlib import Path
from typing import Union, Tuple
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u
import numpy as np
import pandas as pd
from os import listdir
import h5py

from VSPEC.config import RAW_PHOENIX_PATH, BINNED_PHOENIX_PATH
from VSPEC import config
from VSPEC.spectra import get_wavelengths, bin_spectra

WL_UNIT_NEXTGEN = u.AA
FL_UNIT_NEXGEN = u.Unit('erg cm-2 s-1 cm-1')


def get_binned_options():
    """
    Get the available resolving powers of pre-binned spectra.

    Returns
    -------
    list of int
        The available resolving powers.
    """
    dirs = listdir(BINNED_PHOENIX_PATH)
    resolving_powers = np.array([int(dir[2:]) for dir in dirs])
    return np.sort(resolving_powers,)


class RawReader:
    _path = RAW_PHOENIX_PATH
    _teff_unit = config.teff_unit
    _wl_unit_model = WL_UNIT_NEXTGEN
    _fl_unit_model = FL_UNIT_NEXGEN

    @staticmethod
    def get_filename(teff: u.Quantity) -> str:
        """
        Get the filename for a raw PHOENIX model.

        Parameters
        ----------
        teff : astropy.units.Quantity
            The effective temperature of the model

        Returns
        -------
        str
            The filename of the model.
        """
        return f'lte0{teff.to_value(config.teff_unit):.0f}-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'

    def read(self, teff: u.Quantity):
        """
        Read a raw PHOENIX model.

        Parameters
        ----------
        teff : astropy.units.Quantity
            The effective temperature of the model.

        Returns
        -------
        wl : astropy.units.Quantity
            The wavelength axis of the model.
        fl : astropy.units.Quantity
            The flux values of the model.
        """
        fh5 = h5py.File(self._path/self.get_filename(teff), 'r')
        wl = fh5['PHOENIX_SPECTRUM/wl'][()] * self._wl_unit_model
        fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()] * self._fl_unit_model
        wl = wl.to(config.wl_unit)
        fl = fl.to(config.flux_unit)
        return wl, fl


class BinnedReader:
    _path = BINNED_PHOENIX_PATH
    _teff_unit = config.teff_unit
    _wl_unit_model = config.wl_unit
    _fl_unit_model = config.flux_unit

    @staticmethod
    def get_dirname(R: int) -> str:
        """
        Get the name of the directory with a resolving power ``R``.

        Parameters
        ----------
        R : int
            The resolving power.

        Returns
        -------
        str
            The name of the directory
        """
        return f'R_{R:0>6}'

    @staticmethod
    def get_filename(teff: u.Quantity) -> str:
        """
        Get the filename of a binned PHOENIX model.

        Parameters
        ----------
        teff : astropy.units.Quantity
            The effective temperature of the model.

        Returns
        -------
        str
            The filename of the model.
        """
        return f'binned{teff.to_value(config.teff_unit):.0f}StellarModel.txt'

    def read(self, R: int, teff: u.Quantity):
        """
        Read a binned PHOENIX model spectrum

        Parameters
        ----------
        R : int
            The resolving power
        teff : astropy.units.Quantity
            The effective temperature

        Returns
        -------
        wl : astropy.units.Quantity
            The wavelength axis of the model.
        fl : astropy.units.Quantity
            The flux values of the model.
        """
        path = self._path / self.get_dirname(R) / self.get_filename(teff)
        data = pd.read_csv(path)
        wave_col = data.columns[0]
        flux_col = data.columns[1]
        wave_unit_str = wave_col.split('[')[1][:-1]
        flux_unit_str = flux_col.split('[')[1][:-1]
        wl = data[wave_col].values * u.Unit(wave_unit_str)
        fl = data[flux_col].values * u.Unit(flux_unit_str)
        wl = wl.to(config.wl_unit)
        fl = fl.to(config.flux_unit)
        return wl, fl


def read_phoenix(
    teff: u.Quantity,
    R: int,
    w1: u.Quantity,
    w2: u.Quantity
) -> Tuple[u.Quantity, u.Quantity]:
    """
    Read a PHOENIX model and return an appropriately binned version

    Parameters
    ----------
    teff : astropy.units.Quantity
        The effective temperature of the model.
    R : int
        The desired resolving power.
    w1 : astropy.units.Quantity
        The blue wavelength limit.
    w2 : astropy.units.Quantity
        The red wavelenght limit.

    Returns
    -------
    wl_new : astropy.units.Quantity
            The wavelength axis of the model.
    fl_new : astropy.units.Quantity
        The flux values of the model.
    """
    binned_options = get_binned_options()
    options_gte = binned_options >= R
    if not np.any(options_gte):
        wl, flux = RawReader().read(teff)
    else:
        binned_R = np.min(binned_options[options_gte])
        wl, flux = BinnedReader().read(binned_R, teff)
    wl_new: u.Quantity = get_wavelengths(R, w1.to_value(
        config.wl_unit), w2.to_value(config.wl_unit))*config.wl_unit
    try:
        fl_new = bin_spectra(
            wl_old=wl.to_value(config.wl_unit),
            fl_old=flux.to_value(config.flux_unit),
            wl_new=wl_new.to_value(config.wl_unit)
        )*config.flux_unit
    except ValueError:
        interp = RegularGridInterpolator(
            points=[wl.to_value(config.wl_unit)],
            values=flux.to_value(config.flux_unit)
        )
        fl_new = interp(wl_new.to_value(config.wl_unit))*config.flux_unit
    return wl_new[:-1], fl_new


def write_binned_spectrum(
    path: Path,
    wavelength: u.Quantity,
    flux: u.Quantity
):
    """
    Write a binned spectrum to file.

    Parameters
    ----------
    path : pathlib.Path
        The path to write the file to.
    wavelength : astropy.units.Quantity
        The wavelength values to write.
    flux : astropy.units.Quantity
        The flux values to write.
    """
    if not path.parent.exists():
        path.parent.mkdir()
    with open(path, 'w', encoding='UTF-8') as file:
        wavelength_unit_str = str(wavelength.unit)
        flux_unit_str = str(flux.unit)
        file.write(f'wavelength[{wavelength_unit_str}], flux[{flux_unit_str}]')
        for wl, fl in zip(wavelength.value, flux.value):
            file.write(f'\n{wl:.6e}, {fl:.6e}')
