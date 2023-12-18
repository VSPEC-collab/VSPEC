"""VSPEC analysis module

This module is designed to allow a user to easily handle
`VSPEC` outputs.
"""

from typing import Dict
from copy import deepcopy
from pathlib import Path
import warnings
import numpy as np
from astropy import units as u
from astropy.io import fits
from datetime import datetime
import json
from astropy.table import QTable
from pypsg import PyLyr

from VSPEC.config import N_ZFILL, MOLEC_DATA_PATH


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
    layers : astropy.io.fits.HDUList
        `HDUList` of layer arrays
    """

    def __init__(self, path, fluxunit=u.Unit('W m-2 um-1')):
        if not isinstance(path, Path):
            path = Path(path)
        self.observation_data: QTable = QTable.read(path / 'observation_info.fits')
        self.N_images = len(self.observation_data)
        self.time = self.observation_data['time']
        self.phase = self.observation_data['phase']
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
            filename = path / f'phase{str(i).zfill(N_ZFILL)}.fits'
            spectra: QTable = QTable.read(filename)
            if i == 0:  # only do once
                # wavelength
                col = 'wavelength'
                self.wavelength = spectra[col]
            # star
            col = 'star'
            star.append((spectra[col]).to_value(fluxunit))
            # reflected
            col = 'reflected'
            reflected.append((spectra[col]).to_value(fluxunit))
            # reflected
            col = 'planet_thermal'
            thermal.append((spectra[col]).to_value(fluxunit))
            # total
            col = 'total'
            total.append((spectra[col]).to_value(fluxunit))
            # noise
            col = 'noise'
            noise.append((spectra[col]).to_value(fluxunit))
        self.star = np.asarray(star).T * fluxunit
        self.reflected = np.asarray(reflected).T * fluxunit
        self.thermal = np.asarray(thermal).T * fluxunit
        self.total = np.asarray(total).T * fluxunit
        self.noise = np.asarray(noise).T * fluxunit

        try:
            first_lyr:QTable = PyLyr.from_fits(path / f'layer{str(0).zfill(N_ZFILL)}.fits').prof
            colnames = first_lyr.colnames
            vartables: Dict[str, QTable] = {}
            for name in colnames:
                tab = QTable()
                vartables[name] = tab
            
            for i in range(self.N_images):
                filename = path / f'layer{str(i).zfill(N_ZFILL)}.fits'
                dat:QTable = PyLyr.from_fits(filename).prof
                for name in colnames:
                    val = dat[name].value
                    unit = dat[name].unit
                    colname=f'col{i}'
                    vartables[name].add_column(val*unit, name=colname)
            
            self.layers = vartables
        except FileNotFoundError:
            warnings.warn(
                'No Layer info, maybe globes or molecular signatures are off', RuntimeWarning)
            self.layers = fits.HDUList([])

    def get_mean_molecular_mass(self):
        """
        Get the mean molecular mass
        """
        with open(MOLEC_DATA_PATH, 'rt',encoding='UTF-8') as file:
            molec_data = json.loads(file.read())
        shape = self.get_layer('Alt').shape
        mean_molec_mass = np.zeros(shape=shape)*u.g/u.mol
        for mol, dat in molec_data.items():
            mass = dat['mass']
            try:
                data = self.get_layer(mol)
                mean_molec_mass += data*mass*u.g/u.mol
            except KeyError:
                pass
        return mean_molec_mass
        

    def get_layer(self, var: str) -> u.Quantity:
        """
        Get data from layer variable.

        Access the `self.layers` attribute and return the result as
        a `astropy.units.Quantity` object for a single variable.

        Parameters
        ----------
        var : str
            The name of the variable to access

        Returns
        -------
        astropy.units.Quantity
            They layering data of the requested variable

        Raises
        ------
        KeyError
            If `self` does not have any image data or if `var` is not recognized
        """
        if len(self.layers) == 0:
            raise KeyError('`self.layers` does not contain any data')
        if var == 'MEAN_MASS':
            return self.get_mean_molecular_mass()
        tab: QTable = self.layers[var]
        cols = tab.colnames
        unit = tab[cols[0]].unit
        return np.array(
            [tab[col].to_value(unit) for col in cols],
        )*unit

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
        flux:np.ndarray = getattr(self, source)[pixel, :]
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
            flux = (flux/flux[normalize]).to_value(u.dimensionless_unscaled)
        elif isinstance(normalize, str):
            if normalize == 'max':
                flux = (flux/flux.max()).to_value(u.dimensionless_unscaled)
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

        Parameters
        ----------
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

    def to_fits(self) -> fits.HDUList:
        """
        To Fits

        Covert `PhaseAnalyzer` to an 
        `astropy.io.fits.HDUList` object

        Returns
        -------
        astropy.io.fits.HDUList
            Data converted to the `.fits` format
        """
        primary = fits.PrimaryHDU()
        primary.header['CREATED'] = datetime.now().strftime('%Y%m%d-%H%M%S%Z')
        primary.header['N_images'] = self.N_images

        cols = []
        for col in self.observation_data.colnames:
            cols.append(fits.Column(
                name=col, array=self.observation_data[col].values, format='D'))
        obs_tab = fits.BinTableHDU.from_columns(cols)
        obs_tab.name = 'OBS'

        time = fits.Column(name='time', array=self.time.value, format='D')
        phase = fits.Column(name='phase', array=self.phase.value, format='D')
        unique_phase = fits.Column(
            name='unique_phase', array=self.unique_phase.value, format='D')
        tab1 = fits.BinTableHDU.from_columns([time, phase, unique_phase])
        tab1.header['U_TIME'] = str(self.time.unit)
        tab1.header['U_PHASE'] = str(self.phase.unit)
        tab1.header['U_UPHASE'] = str(self.unique_phase.unit)
        tab1.name = 'PHASE'
        wavelength = fits.Column(
            name='wavelength', array=self.wavelength.value, format='D')
        tab2 = fits.BinTableHDU.from_columns([wavelength])
        tab2.header['U_WAVE'] = str(self.wavelength.unit)
        tab2.name = 'WAVELENGTH'

        total = fits.ImageHDU(self.total.value)
        total.header['U_FLUX'] = str(self.total.unit)
        total.name = 'TOTAL'
        star = fits.ImageHDU(self.star.value)
        star.header['U_FLUX'] = str(self.star.unit)
        star.name = 'STAR'
        reflected = fits.ImageHDU(self.reflected.value)
        reflected.header['U_FLUX'] = str(self.reflected.unit)
        reflected.name = 'REFLECTED'
        thermal = fits.ImageHDU(self.thermal.value)
        thermal.header['U_FLUX'] = str(self.thermal.unit)
        thermal.name = 'THERMAL'
        noise = fits.ImageHDU(self.noise.value)
        noise.header['U_FLUX'] = str(self.noise.unit)
        noise.name = 'NOISE'

        hdul = fits.HDUList([primary, obs_tab, tab1, tab2,
                            total, star, reflected, thermal, noise])
        hdul = hdul + self.layers
        return fits.HDUList(hdul)

    def write_fits(self, filename: str) -> None:
        """
        Save `PhaseAnalyzer` object as a `.fits` file.

        Parameters
        ----------
        filename : str
        """
        hdul = self.to_fits()
        hdul.writeto(filename,overwrite=True)
    def to_twocolumn(self,index:tuple,outfile:str,fmt='ppm',wl='um'):
        """
        Write data to a two column file that can be used in a retrival.
        """
        if fmt == 'ppm':
            flux = (self.spectrum('thermal',index,False)+self.spectrum('reflected',index,False))/self.spectrum('total',index,False) * 1e6
            noise = self.spectrum('noise',index,False)/self.spectrum('total',index,False) * 1e6
            flux_unit = u.dimensionless_unscaled
        elif fmt == 'flambda':
            flux = self.spectrum('thermal',index,False)+self.spectrum('reflected',index,False)
            noise = self.spectrum('noise',index,False)
            flux_unit = u.Unit('W m-2 um-1')
        else:
            raise ValueError(f'Unknown format "{fmt}"')
        wl_unit = u.Unit(wl)
        wl = self.wavelength.to_value(wl_unit)
        flux = flux.to_value(flux_unit)
        noise = noise.to_value(flux_unit)
        with open(outfile,'wt',encoding='ascii') as file:
            for w,f,n in zip(wl,flux,noise):
                file.write(f'{w:<10.4f}{f:<14.4e}{n:<14.4e}\n')

