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
from astropy import units as u, constants as c
from astropy.io import fits
from datetime import datetime
import json

from VSPEC.helpers import to_float
from VSPEC.files import N_ZFILL, MOLEC_DATA_PATH


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
            layer_data = np.array(layers)
            hdus = []
            for i, var in enumerate(cols):
                dat = layer_data[:, :, i]
                if '[' in var:
                    unit = u.Unit(var.split('[')[1].replace(']', ''))
                    var = var.split('[')[0]
                else:
                    unit = u.dimensionless_unscaled
                image = fits.ImageHDU(dat)
                image.header['AXIS0'] = 'PHASE'
                image.header['AXIS1'] = 'LAYER'
                image.header['VAR'] = var
                image.header['UNIT'] = str(unit)
                image.name = var
                hdus.append(image)
            self.layers = fits.HDUList(hdus)
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
        hdu = self.layers[var]
        unit = u.Unit(hdu.header['UNIT'])
        return hdu.data*unit

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
        for col in self.observation_data.columns:
            cols.append(fits.Column(
                name=col, array=self.observation_data[col].value, format='K'))
        obs_tab = fits.BinTableHDU.from_columns(cols)
        obs_tab.name = 'OBS'

        time = fits.Column(name='time', array=self.time.value, format='K')
        phase = fits.Column(name='phase', array=self.phase.value, format='K')
        unique_phase = fits.Column(
            name='unique_phase', array=self.unique_phase.value, format='K')
        tab1 = fits.BinTableHDU.from_columns([time, phase, unique_phase])
        tab1.header['TIME_UNIT'] = str(self.time.unit)
        tab1.header['PHASE_UNIT'] = str(self.phase.unit)
        tab1.header['UNIQUE_PHASE_UNIT'] = str(self.unique_phase.unit)
        tab1.name = 'PHASE'
        wavelength = fits.Column(
            name='wavelength', array=self.wavelength.value, format='K')
        tab2 = fits.BinTableHDU.from_columns([wavelength])
        tab2.header['WAVELENGTH_UNIT'] = str(self.wavelength.unit)
        tab2.name = 'WAVELENGTH'

        total = fits.ImageHDU(self.total.value)
        total.header['FLUX_UNIT'] = str(self.total.unit)
        total.name = 'TOTAL'
        star = fits.ImageHDU(self.star.value)
        star.header['FLUX_UNIT'] = str(self.star.unit)
        star.name = 'STAR'
        reflected = fits.ImageHDU(self.reflected.value)
        reflected.header['FLUX_UNIT'] = str(self.reflected.unit)
        reflected.name = 'REFLECTED'
        thermal = fits.ImageHDU(self.thermal.value)
        thermal.header['FLUX_UNIT'] = str(self.thermal.unit)
        thermal.name = 'THERMAL'
        noise = fits.ImageHDU(self.noise.value)
        noise.header['FLUX_UNIT'] = str(self.noise.unit)
        noise.name = 'NOISE'

        hdul = fits.HDUList([primary, obs_tab, tab1, tab2,
                            total, star, reflected, thermal, noise])
        hdul = hdul + self.layers
        return hdul

    def write_fits(self, filename: str) -> None:
        """
        Save `PhaseAnalyzer` object as a `.fits` file.

        Parameters
        ----------
        filename : str
        """
        hdul = self.to_fits()
        hdul.writeto(filename)


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
    if len(lines) == 0:
        raise ValueError('No data was captured. Perhaps the format is wrong.')
    dat = StringIO('\n'.join(lines[1:]))
    names = lines[0].split()
    for i, name in enumerate(names):
        # get previous parameter (e.g 'water' for 'water_size')
        if 'size' in name:
            names[i] = names[i-1] + '_' + name
    return pd.read_csv(dat, delim_whitespace=True, names=names)


def get_gcm_binary(filename):
    key = '<ATMOSPHERE-GCM-PARAMETERS>'
    start = b'<BINARY>'
    end = b'</BINARY>'
    with open(filename,'rb') as file:
        fdat = file.read()
    header, dat = fdat.split(start)
    dat = dat.replace(end,b'')
    dat = np.frombuffer(dat,dtype='float32')
    for line in str(header).split(r'\n'):
        if key in line:
            return line.replace(key,''),np.array(dat)
def sep_header(header):
    fields = header.split(',')
    coords = fields[:7]
    var = fields[7:]
    return coords,var


class GCMdecoder:
    DOUBLE = ['Winds']
    FLAT = ['Tsurf','Psurf','Albedo','Emissivity']
    def __init__(self,header,dat):
        self.header=header
        self.dat=dat
    @classmethod
    def from_psg(cls,filename):
        head,dat = get_gcm_binary(filename)
        return cls(head,dat)
    def get_shape(self):
        coord,_ = sep_header(self.header)
        Nlon,Nlat,Nlayer, _,_,_,_ = coord
        return int(Nlon),int(Nlat),int(Nlayer)
    def get_3d_size(self):
        Nlon,Nlat,Nlayer = self.get_shape()
        return Nlon*Nlat*Nlayer
    def get_2d_size(self):
        Nlon,Nlat,_ = self.get_shape()
        return Nlon*Nlat
    def get_lats(self):
        coord,_ = sep_header(self.header)
        _,Nlat,_,_,lat0,_,dlat = coord
        return np.arange(int(Nlat))*float(dlat) + float(lat0)
    def get_lons(self):
        coord,_ = sep_header(self.header)
        Nlon,_,_,lon0,_,dlon,_ = coord
        return np.arange(int(Nlon))*float(dlon) + float(lon0)
    def __getitem__(self,item):
        _, variables = sep_header(self.header)
        if not item in variables:
            raise KeyError(f'{item} not found. Acceptable keys are {variables}')
        else:
            start = 0
            def get_array_length(var):
                if var in self.DOUBLE:
                    return 2*self.get_3d_size(), 'double'
                elif var in self.FLAT:
                    return self.get_2d_size(), 'flat'
                else:
                    return self.get_3d_size(), 'single'
            def package_array(dat,key):
                if key == 'single':
                    Nlat,Nlon,Nlayer = self.get_shape()
                    return dat.reshape(Nlayer,Nlon,Nlat)
                elif key == 'flat':
                    Nlat,Nlon,Nlayer = self.get_shape()
                    return dat.reshape(Nlon,Nlat)
                elif key == 'double':
                    Nlat,Nlon,Nlayer = self.get_shape()
                    return dat.reshape(2,Nlayer,Nlon,Nlat)
                else:
                    raise ValueError(f'Unknown value {key}')
            for var in variables:
                size,key = get_array_length(var)
                if item==var:
                    dat = self.dat[start:start+size]
                    return package_array(dat,key)
                else:
                    start+=size
    def __setitem__(self,item:str,new_value:np.ndarray):
        """
        set an array
        """
        old_value = self.__getitem__(item)
        if not old_value.shape == new_value.shape:
            raise ValueError('New shape must match old shape.')
        new_value = new_value.astype(old_value.dtype)
        _, variables = sep_header(self.header)
        def get_array_length(var):
            if var in self.DOUBLE:
                return 2*self.get_3d_size(), 'double'
            elif var in self.FLAT:
                return self.get_2d_size(), 'flat'
            else:
                return self.get_3d_size(), 'single'
        start = 0
        for var in variables:
            size,_ = get_array_length(var)
            if item==var:
                dat = new_value.flatten(order='C')
                self.dat[start:start+size] = dat
                return None
            else:
                start+=size
    def copy_config(self,path_to_copy:Path,path_to_write:Path):
        """
        Copy a PSG config file but overwrite all GCM parameters and data
        """
        with open(path_to_copy,'rb') as infile:
            with open(path_to_write, 'wb') as outfile:
                contents = infile.read()
                t,b = contents.split(b'<BINARY>')
                b = b.replace(b'</BINARY>',b'')
                lines = t.split(b'\n')
                for line in lines:
                    if b'<ATMOSPHERE-GCM-PARAMETERS>' in line:
                        outfile.write(bytes(self.header + '\n',encoding='UTF-8'))
                    else:
                        outfile.write(line + b'\n')
                outfile.write(b'<BINARY>')
                outfile.write(np.asarray(self.dat,dtype='float32',order='C'))
                outfile.write(b'</BINARY>')
        

    def get_mean_molec_mass(self):
        """
        Get the mean molecular mass at every point on the GCM
        """
        with open(MOLEC_DATA_PATH, 'rt',encoding='UTF-8') as file:
            molec_data = json.loads(file.read())
        Nlon,Nlat,Nlayer = self.get_shape()
        mean_molec_mass = np.zeros(shape=(Nlayer,Nlat,Nlon))*u.g/u.mol
        for mol, dat in molec_data.items():
            mass = dat['mass']
            try:
                data = self[mol]
                mean_molec_mass += data*mass*u.g/u.mol
            except KeyError:
                pass
        return mean_molec_mass
    def get_alt(self,M:u.Quantity,R:u.Quantity):
        """
        Get the altitude of each GCM point.
        """
        P = 10**self['Pressure']*u.bar
        T = self['Temperature']*u.K
        m = self.get_mean_molec_mass()
        Nlon, Nlat, Nlayers = self.get_shape()
        z_unit = u.km
        z = [np.zeros(shape=(Nlat,Nlon))]
        for i in range(Nlayers-1):
            dP = P[i+1,:,:] - P[i,:,:]
            rho = m[i,:,:]*(P[i,:,:]+ 0.5*dP)/c.R/T[i,:,:]
            r = z[-1]*z_unit + R
            g = M*c.G/r**2
            dz = -dP/rho/g
            z.append((z[-1]*z_unit+dz).to(z_unit).value)
        return z*z_unit
    def get_column_density(self,mol:str,M:u.Quantity,R:u.Quantity,):
        """
        Get the column density of a gas at each point on the gcm.
        """
        abn = self[mol]*u.mol/u.mol
        P = 10**self['Pressure']*u.bar
        T = self['Temperature']*u.K
        partial_pressure = P*abn
        alt = self.get_alt(M,R)
        heights = np.diff(alt,axis=0)
        density = np.sum(partial_pressure[:-1]*heights/c.R/T[:-1],axis=0)
        return density.to(u.mol/u.cm**2)

    def get_column_clouds(self,var:str,M:u.Quantity,R:u.Quantity,):
        """
        Get the column density of a cloud at each point on the gcm.
        """
        mass_frac = 10**self[var]*u.kg/u.kg
        P = 10**self['Pressure']*u.bar
        T = self['Temperature']*u.K
        molar_mass = self.get_mean_molec_mass()
        alt = self.get_alt(M,R)
        heights = np.diff(alt,axis=0)
        gas_mass_density = P[:-1]*heights/c.R/T[:-1]*molar_mass[:-1] # g cm-2
        mass_density = np.sum(mass_frac[:-1]*gas_mass_density,axis=0).cgs
        return mass_density.to(u.kg/u.cm**2)
