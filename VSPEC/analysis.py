"""VSPEC analysis module

This module is designed to allow a user to easily handle
`VSPEC` outputs.
"""

import re
from copy import deepcopy
from io import StringIO
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from astropy import units as u
import xarray

from VSPEC.helpers import to_float
from VSPEC.files import N_ZFILL


class PhaseAnalyzer:
    """Class to store and analyze `VSPEC` phase curves

    Class to read all the data produced from a phase `VSPEC` curve simulation.
    This class also includes some basic-but-powerfull analysis methods meant to be
    quickly used to create nice figures.

    Parameters
    ----------
    path : pathlib.Path or str
        The path to the directory storing all the final output.
        This is usually `.../AllModelSpectraValues/`
    fluxunit : astropy.units.Unit, default=u.Unit('W m-2 um-1')
        Standard unit to use with flux values. This way they are safely
        converted between `Quantity` and `float`

    Attributes
    ----------
    observation_data : pandas.DataFrame
        DataFrame containing the observation geometry at each epoch.
    N_images : int
        Number of epochs in observation
    time : astropy.units.Quantity
        Time coordinate of each epoch
    phase : astropy.units.Quantity
        Phase of the planet at each epoch. Between 0 and 360 degrees
    unique_phase : astropy.units.Quantity
        Non-cyclical phase of the planet at each epoch.
        Can be greater than 360 degrees
    wavelength : astropy.units.Quantity
        Wavelength values of the spectral axis.
    star : astropy.units.Quantity
        2D array of stellar flux
    reflected : astropy.units.Quantity
        2D array of reflected flux
    thermal : astropy.units.Quantity
        2D array of thermal flux
    total : astropy.units.Quantity
        2D array of total flux
    noise : astropy.units.Quantity
        2D array of noise flux
    layers : xarray.DataArray
        3D DataArray of Layer data
    """

    def __init__(self, path, fluxunit=u.Unit('W m-2 um-1')):
        if not isinstance(path, Path):
            path = Path(path)
        self.observation_data = pd.read_csv(path / 'observation_info.csv')
        self.N_images = len(self.observation_data)
        self.time = self.observation_data['time[s]'].values * u.s
        self.phase = self.observation_data['phase[deg]'].values * u.deg
        self.unique_phase = deepcopy(self.phase)
        for i in range(len(self.unique_phase) - 1):
            while self.unique_phase[i] > self.unique_phase[i+1]:
                self.unique_phase[i+1] += 360*u.deg
        star = []
        # star_facing_planet = pd.DataFrame() # not super interesting
        reflected = []
        thermal = []
        total = []
        noise = []
        for i in range(self.N_images):
            filename = path / f'phase{str(i).zfill(N_ZFILL)}.csv'
            spectra = pd.read_csv(filename)
            cols = pd.Series(spectra.columns)
            if i == 0:  # only do once
                # wavelength
                col = cols[cols.str.contains('wavelength')]
                unit = u.Unit(re.findall(r'\[([\w\d\/ \(\)]+)\]', col[0])[0])
                self.wavelength = spectra[col].values.T[0] * unit
            # star
            col = cols[cols.str.contains(r'star\[')].values[0]
            unit = u.Unit(re.findall(r'\[([\w\d\/ \(\)]+)\]', col)[0])
            star.append(to_float(spectra[col].values * unit, fluxunit))
            # reflected
            col = cols[cols.str.contains(r'reflected\[')].values[0]
            unit = u.Unit(re.findall(r'\[([\w\d\/ \(\)]+)\]', col)[0])
            reflected.append(to_float(spectra[col].values * unit, fluxunit))
            # reflected
            col = cols[cols.str.contains(r'planet_thermal\[')].values[0]
            unit = u.Unit(re.findall(r'\[([\w\d\/ \(\)]+)\]', col)[0])
            thermal.append(to_float(spectra[col].values * unit, fluxunit))
            # total
            col = cols[cols.str.contains(r'total\[')].values[0]
            unit = u.Unit(re.findall(r'\[([\w\d\/ \(\)]+)\]', col)[0])
            total.append(to_float(spectra[col].values * unit, fluxunit))
            # noise
            col = cols[cols.str.contains(r'noise\[')].values[0]
            unit = u.Unit(re.findall(r'\[([\w\d\/ \(\)]+)\]', col)[0])
            noise.append(to_float(spectra[col].values * unit, fluxunit))
        self.star = np.asarray(star).T * fluxunit
        self.reflected = np.asarray(reflected).T * fluxunit
        self.thermal = np.asarray(thermal).T * fluxunit
        self.total = np.asarray(total).T * fluxunit
        self.noise = np.asarray(noise).T * fluxunit

        try:
            layers = []
            first = True
            for i in range(self.N_images):
                filename = path / f'layer{str(i).zfill(N_ZFILL)}.csv'
                dat = pd.read_csv(filename)
                if not first:
                    assert np.all(dat.columns == cols)
                else:
                    first = False
                cols = dat.columns
                layers.append(dat.values)
            index = np.arange(layers[0].shape[0])
            self.layers = xarray.DataArray(np.array(layers), dims=['phase', 'layer', 'var'], coords={
                                           'phase': self.unique_phase, 'layer': index, 'var': cols})
        except FileNotFoundError:
            warnings.warn(
                'No Layer info, maybe globes or molecular signatures are off', RuntimeWarning)
            self.layers = None

    def lightcurve(self, source, pixel, normalize='none', noise=False):
        """
        Produce a lightcurve

        Return the lightcurve of `source` of the wavelengths described by `pixel`

        Parameters
        ----------
        source : str
            Which data array to access.
        pixel : int or 2-tuple of int
            Pixel(s) of spectral axis to use when building lightcurve.
        normalize : str or int, default='none'
            Normalization scheme. If integer, pixel of time axis to
            normalize the lightcurve to. Otherwise it is a keyword
            to describe the normalization process: `'none'` or `'max'`
        noise : bool or float or int, default=False
            Should gaussian noise be added? If float, scale gaussian noise
            by this parameter.

        Returns
        -------
        astropy.units.Quantity
            Lightcurve of the desired source in the desired bandpass

        Raises
        ------
        ValueError
            If `noise` is not `bool`, `float`, or `int`
        ValueError
            If `normalize` is not recognized or `True`

        Warns
        -----
        RuntimeWarning
            If `normalize` is `False`

        """
        if isinstance(pixel, tuple):
            pixel = slice(*pixel)
        flux = getattr(self, source)[pixel, :]
        if isinstance(noise, bool):
            if noise:
                flux = flux + \
                    np.random.normal(
                        scale=self.noise.value[pixel, :])*self.noise.unit
        elif isinstance(noise, float) or isinstance(noise, int):
            flux = flux + noise * \
                np.random.normal(
                    scale=self.noise.value[pixel, :])*self.noise.unit
        else:
            raise ValueError('noise parameter must be bool, float, or int')
        if flux.ndim > 1:
            flux = flux.mean(axis=0)
        if isinstance(normalize, int):
            flux = to_float(flux/flux[normalize], u.Unit(''))
        elif isinstance(normalize, str):
            if normalize == 'max':
                flux = to_float(flux/flux.max(), u.Unit(''))
            elif normalize == 'none':
                pass
            else:
                raise ValueError(f'Unknown normalization scheme: {normalize}')
        elif isinstance(normalize, bool):
            if normalize:
                raise ValueError(
                    '`normalize=True` is ambiguous. Please specify a valid scheme')
            else:
                message = 'Setting `normalize=False` can be dangerous as `True` is an ambigous value. '
                message += 'Please use `normalize="none"` instead.'
                warnings.warn(message, RuntimeWarning)
        else:
            raise ValueError(f'Unknown normalization parameter: {normalize}')
        return flux

    def spectrum(self, source, images, noise=False):
        """
        Get a 1D spectrum

        Return the spectrum of a specified source at a single epoch
        or average over multiple epochs.

        Parameters:
        -----------
        source : str
            Which data array to access. If `'noise'` is specified, use propagation
            of error formula to calculate theoretical noise of spectrum.
        images : int or 2-tuple of int
            Pixel(s) of time axis to use when building spectrum.
        noise : bool or float or int
            Should gaussian noise be added? If float, scale gaussian noise
            by this parameter.

        Returns
        -------
        astropy.units.Quantity
            Spectrum of the desired source over the desired epoch(s)

        Raises
        ------
        ValueError
            If `noise` is not `bool`, `float`, or `int`
        """
        if isinstance(images, tuple):
            images = slice(*images)
        if source == 'noise':
            flux = self.noise[:, images]**2
            try:
                n_images = flux.shape[1]
                flux = flux.sum(axis=1)/n_images**2
            except IndexError:
                pass
            return np.sqrt(flux)
        else:
            flux = getattr(self, source)[:, images]
            if isinstance(noise, bool):
                if noise:
                    flux = flux + \
                        np.random.normal(
                            scale=self.noise.value[:, images])*self.noise.unit
            elif isinstance(noise, float) or isinstance(noise, int):
                flux = flux + noise * \
                    np.random.normal(
                        scale=self.noise.value[:, images])*self.noise.unit
            else:
                raise ValueError('noise parameter must be bool, float, or int')
            try:
                flux = flux.mean(axis=1)
            except IndexError:  # images is int
                pass
            return flux


def read_lyr(filename: str) -> pd.DataFrame:
    """
    Read layer file

    Parse a PSG .lyr file and turn it into a
    pandas DataFrame.

    Parameters
    ----------
    filename : str
        The name of the layer file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the layer data.
    """
    lines = []
    with open(filename, 'r', encoding='UTF-8') as file:
        save = False
        for line in file:
            if 'Alt[km]' in line:
                save = True
            if save:
                if '--' in line:
                    if len(lines) > 2:
                        save = False
                    else:
                        pass
                else:
                    lines.append(line[2:-1])
    if len(lines)==0:
        raise ValueError('No data was captured. Perhaps the format is wrong.')
    dat = StringIO('\n'.join(lines[1:]))
    names = lines[0].split()
    for i, name in enumerate(names):
        # get previous parameter (e.g 'water' for 'water_size')
        if 'size' in name:
            names[i] = names[i-1] + '_' + name
    return pd.read_csv(dat, delim_whitespace=True, names=names)
