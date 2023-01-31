import re
from copy import deepcopy

import numpy as np
import pandas as pd
from astropy import units as u
from io import StringIO
import xarray

from VSPEC.helpers import to_float


class PhaseAnalyzer:
    """Phase Analyzer
    Class to read all the data produced from a phase-resolved simulation

    Args:
        path (pathlib.Path): path containing the combined spectra. Probably
            `.../Data/AllModelSpectraValues`
        fluxunit (astropy.units.quantity.Quantity [flux]): unit of flux to standardize the dataset

    Returns:
        None
    """
    def __init__(self,path,fluxunit = u.Unit('W m-2 um-1')):
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
            filename = path / f'phase{str(i).zfill(3)}.csv'
            spectra = pd.read_csv(filename)
            cols = pd.Series(spectra.columns)
            if i==0: # only do once
                # wavelength
                col = cols[cols.str.contains('wavelength')]
                unit = u.Unit(re.findall('\[([\w\d\/ \(\)]+)\]', col[0])[0])
                self.wavelength = spectra[col].values.T[0] * unit
            # star
            col = cols[cols.str.contains('star\[')].values[0]
            unit = u.Unit(re.findall('\[([\w\d\/ \(\)]+)\]', col)[0])
            star.append(to_float(spectra[col].values * unit,fluxunit))
            # reflected
            col = cols[cols.str.contains('reflected\[')].values[0]
            unit = u.Unit(re.findall('\[([\w\d\/ \(\)]+)\]', col)[0])
            reflected.append(to_float(spectra[col].values * unit,fluxunit))
            # reflected
            col = cols[cols.str.contains('planet_thermal\[')].values[0]
            unit = u.Unit(re.findall('\[([\w\d\/ \(\)]+)\]', col)[0])
            thermal.append(to_float(spectra[col].values * unit,fluxunit))
            # total
            col = cols[cols.str.contains('total\[')].values[0]
            unit = u.Unit(re.findall('\[([\w\d\/ \(\)]+)\]', col)[0])
            total.append(to_float(spectra[col].values * unit,fluxunit))
            # noise
            col = cols[cols.str.contains('noise\[')].values[0]
            unit = u.Unit(re.findall('\[([\w\d\/ \(\)]+)\]', col)[0])
            noise.append(to_float(spectra[col].values * unit,fluxunit))
        self.star = np.asarray(star).T * fluxunit
        self.reflected = np.asarray(reflected).T * fluxunit
        self.thermal = np.asarray(thermal).T * fluxunit
        self.total = np.asarray(total).T * fluxunit
        self.noise = np.asarray(noise).T * fluxunit
        
        try:
            layers = []
            first = True
            for i in range(self.N_images):
                filename = path / f'layer{str(i).zfill(3)}.csv'
                dat = pd.read_csv(filename)
                if not first:
                    assert np.all(dat.columns == cols)
                else:
                    first = False
                cols = dat.columns
                layers.append(dat.values)
            index = np.arange(layers[0].shape[0])
            self.layers = xarray.DataArray(np.array(layers),dims = ['phase','layer','var'],coords={'phase':self.unique_phase,'layer':index,'var':cols})
        except FileNotFoundError:
            print('No Layer info, maybe globes is off')
            self.layers = None

    def lightcurve(self,source,pixel,normalize='none',noise=False):
        """lightcurve
        Produce a lightcurve of the data

        Args:
            source (str): which data array to access
            pixel (int or 2-tuple of int): pixel(s) of spectral axis to use when building lightcurve
                if tuple is given, it is used as the input to a `slice` object
            xvar (str): variable of x axis to return with lightcurve. Either `'phase'`, `'time'`, or `'pixel'`
            normalize (str or int): if integer, pixel of time axis to normalize the lightcurve to.
                Otherwise it is a keyword to describe the normalization process: `'none'` or `'max'`
            noise (bool): should gaussian noise be added to the lightcurve

        """
        if isinstance(pixel,tuple):
            pixel = slice(*pixel)
        y = getattr(self,source)[pixel,:]
        if noise:
            y = y + np.random.normal(scale=self.noise.value[pixel,:])*self.noise.unit
        if y.ndim > 1:
            y = y.mean(axis=0)
        if isinstance(normalize,int):
            y = to_float(y/y[normalize],u.Unit(''))
        elif str(normalize)=='max':
            y = to_float(y/y.max(),u.Unit(''))
        return y

    def combine(self,source,images,noise=False):
        """combine
        Combined multiple epochs of spectra to improve snr
        If `'noise'` is specified then propogation of error formula is used

        Args:
            source (str): which data array to access
            images (int or 2-tuple of int): pixel(s) of phase axis to use when building lightcurve
                if tuple is given, it is used as the input to a `slice` object
            noise (bool): Whether to add gaussian noise to output. Ignored if source=`'noise'`
        """
        if isinstance(images,tuple):
            images = slice(*images)
        if source=='noise':
            y = self.noise[:,images]**2
            N_images = y.shape[1]
            y = y.sum(axis=1)/N_images**2
            return np.sqrt(y)
        else:
            y = getattr(self,source)[:,images]
            if noise:
                y = y + np.random.normal(scale=self.noise.value[:,images])*self.noise.unit
            y = y.mean(axis=1)
            return y


def read_lyr(filename:str)->pd.DataFrame:
    """ read layer file

    Parse a PSG .lyr file

    Args:
        filename (str): path to the file
    
    Returns:
        (pd.DataFrame): A DataFrame constructed from the file contents
    """
    lines = []
    with open(filename,'r') as f:
        save = False
        for line in f:
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
    dat = StringIO('\n'.join(lines[1:]))
    names = lines[0].split()
    for i in range(len(names)):
        if 'size' in names[i]:
            names[i] = names[i-1] + '_' + names[i]
    return pd.read_csv(dat,delim_whitespace=True,names = names)
