"""VSPEC stellar spectra module

This module allows `VSPEC` to read and
write model stellar spectra.
"""

from pathlib import Path
from typing import Union, Tuple, Callable
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d, RegularGridInterpolator
from astropy import units as u, constants as c
from VSPEC.helpers import to_float, isclose
from VSPEC.files import RAW_PHOENIX_PATH, BINNED_PHOENIX_PATH

PRE_BINNED = (1000,50)

def get_wavelengths(resolving_power: int, lam1: float, lam2: float) -> np.ndarray:
    """
    Get wavelengths

    Get wavelength points given a resolving power and a desired spectral range.
    Provides one more point than PSG, but is otherwise identical (i.e. a PSG
    spectrum will have wavelength points `lam[:-1]`)

    Parameters
    ----------
    resolving_power : int
        Resolving power.
    lam1 : float
        Initial wavelength.
    lam2 : float
        Final wavelength.

    Returns
    -------
    numpy.ndarray
        Wavelength points.
    """
    lam = lam1
    lams = [lam]
    while lam < lam2:
        dlam = lam / resolving_power
        lam = lam + dlam
        lams.append(lam)
    lams = np.array(lams)
    return lams

def fast_bin_raw_data(path: Union[str, Path], resolving_power: int = 50,
                 lam1: u.Quantity = None, lam2: u.Quantity = None,
                 model_unit_wavelength: u.Unit = u.AA,
                 model_unit_flux: u.Unit = u.Unit('erg cm-2 s-1 cm-1'),
                 target_unit_wavelength: u.Unit = u.um,
                 target_unit_flux: u.Unit = u.Unit('W m-2 um-1')
                 ) -> Tuple[u.Quantity, u.Quantity]:
    """
    Bin raw data. Experimental!!

    Read in a raw H5 spectrum and bin it to the desired resolving power.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the model spectrum.
    resolving_power : int, default=50
        Resolving power of the binned spectrum.
    lam1 : astropy.units.Quantity [length], default=None
        Starting wavelength of binned spectrum. Defaults to the
        shortest wavelength in the raw file.
    lam2 : astropy.units.quantity.Quantity [length], default=None
        Ending wavelength of binned spectrum. Defaults to the
        longest wavelength in the raw file.
    model_unit_wavelength : '~astropy.units.Unit' [length], default=u.AA
        Wavelength unit of the model.
    model_unit_flux : astropy.units.Unit [flux], default=u.Unit('erg cm-2 s-1 cm-1')
        Flux unit of the model.
    target_unit_wavelength : astropy.units.Unit [length], default=u.um
        Wavelength unit of the binned spectrum.
    target_unit_flux : astropy.units.Unit [flux], default=u.Unit('W m-2 um-1')
        Flux unit of the binned spectrum.

    Returns
    -------
    binned_wavelength : astropy.units.quantity.Quantity [length]
        Wavelength points of the binned spectrum.
    binned_flux : astropy.units.quantity.Quantity [flux]
        Flux points of the new spectrum.
    """
    fh5 = h5py.File(path, 'r')
    wl_data = fh5['PHOENIX_SPECTRUM/wl']
    if lam1 is None:
        lam1 = np.min(wl_data)*model_unit_wavelength
    if lam2 is None:
        lam2 = np.max(wl_data)*model_unit_wavelength
    wl = wl_data[:]
    region_to_bin = (wl >= lam1.to_value(model_unit_wavelength)) & (wl <= lam2.to_value(model_unit_wavelength))
    wl = (np.array(wl)[region_to_bin]*model_unit_wavelength).to(target_unit_wavelength)
    fl_data = fh5['PHOENIX_SPECTRUM/flux']
    scale = (1*model_unit_flux).to_value(target_unit_flux)
    fl = np.array(fl_data[:])
    fl = fl[region_to_bin]
    fl = np.power(10.,fl)* scale
    binned_wavelengths = get_wavelengths(resolving_power,
                                         to_float(
                                             lam1, target_unit_wavelength),
                                         to_float(lam2, target_unit_wavelength)) * target_unit_wavelength
    bin_number = np.digitize(wl[:-1],binned_wavelengths[:-1]) # bin ids
    dlam = np.diff(wl).to_value(target_unit_wavelength) #width of each hires channel um
    binned_dlam = np.diff(binned_wavelengths).to_value(target_unit_wavelength) #width of each low res channel um

    weights = fl[:-1]*dlam #W m-2 hi res channel
    binned_flux = np.bincount(bin_number-1,weights)
    return binned_wavelengths[:-1], binned_flux/binned_dlam * target_unit_flux

def bin_from_cache(teff:int,
                 resolving_power: int = 50,
                 lam1: u.Quantity = None, lam2: u.Quantity = None,
                 model_unit_wavelength: u.Unit = u.AA,
                 model_unit_flux: u.Unit = u.Unit('erg cm-2 s-1 cm-1'),
                 target_unit_wavelength: u.Unit = u.um,
                 target_unit_flux: u.Unit = u.Unit('W m-2 um-1')):
    """
    Bin from pre-binned spectrum.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the model spectrum.
    resolving_power : int, default=50
        Resolving power of the binned spectrum.
    lam1 : astropy.units.Quantity [length], default=None
        Starting wavelength of binned spectrum. Defaults to the
        shortest wavelength in the raw file.
    lam2 : astropy.units.quantity.Quantity [length], default=None
        Ending wavelength of binned spectrum. Defaults to the
        longest wavelength in the raw file.
    model_unit_wavelength : '~astropy.units.Unit' [length], default=u.AA
        Wavelength unit of the model.
    model_unit_flux : astropy.units.Unit [flux], default=u.Unit('erg cm-2 s-1 cm-1')
        Flux unit of the model.
    target_unit_wavelength : astropy.units.Unit [length], default=u.um
        Wavelength unit of the binned spectrum.
    target_unit_flux : astropy.units.Unit [flux], default=u.Unit('W m-2 um-1')
        Flux unit of the binned spectrum.

    Returns
    -------
    binned_wavelength : astropy.units.quantity.Quantity [length]
        Wavelength points of the binned spectrum.
    binned_flux : astropy.units.quantity.Quantity [flux]
        Flux points of the new spectrum.
    
    Notes
    -----
    This function has the same signature as `bin_raw_data`, allowing it
    to be used interchangably.
    """
    R_to_use = None
    interp_only = False
    for R in sorted(PRE_BINNED):
        if R >= resolving_power:
            if R==resolving_power:
                interp_only = True
            R_to_use = R
            break
    if R_to_use is None:
        raise ValueError('No suitible pre-binned spectrum could be found')
    
    wl,fl = read_binned_spectrum(
        get_binned_filename(teff),
        get_cached_file_dir(R_to_use)
    )
    wl = wl.to(target_unit_wavelength)
    fl = fl.to(target_unit_flux)
    if lam1 is None:
        lam1 = np.min(wl)
    if lam2 is None:
        lam2 = np.max(wl)
    binned_wavelengths = get_wavelengths(resolving_power,
                                         to_float(
                                             lam1, target_unit_wavelength),
                                         to_float(lam2, target_unit_wavelength)) * target_unit_wavelength
    if interp_only:
        binned_flux = interp1d(
            wl.to_value(target_unit_wavelength),
            fl.to_value(target_unit_flux)
        )(binned_wavelengths.to_value(target_unit_wavelength)[:-1])
        return binned_wavelengths[:-1], binned_flux*target_unit_flux
    else:
        region_to_bin = (wl >= lam1) & (wl <= lam2)
        wl = wl[region_to_bin]
        fl = fl[region_to_bin]
        binned_flux = bin_spectra(
            np.array(wl.to_value(target_unit_wavelength)),
            np.array(fl.to_value(target_unit_flux)),
            np.array(binned_wavelengths.to_value(target_unit_wavelength))
        )
        return binned_wavelengths[:-1], binned_flux*target_unit_flux



def bin_raw_data(path: Union[str, Path], resolving_power: int = 50,
                 lam1: u.Quantity = None, lam2: u.Quantity = None,
                 model_unit_wavelength: u.Unit = u.AA,
                 model_unit_flux: u.Unit = u.Unit('erg cm-2 s-1 cm-1'),
                 target_unit_wavelength: u.Unit = u.um,
                 target_unit_flux: u.Unit = u.Unit('W m-2 um-1')
                 ) -> Tuple[u.Quantity, u.Quantity]:
    """
    Bin raw data.

    Read in a raw H5 spectrum and bin it to the desired resolving power.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the model spectrum.
    resolving_power : int, default=50
        Resolving power of the binned spectrum.
    lam1 : astropy.units.Quantity [length], default=None
        Starting wavelength of binned spectrum. Defaults to the
        shortest wavelength in the raw file.
    lam2 : astropy.units.quantity.Quantity [length], default=None
        Ending wavelength of binned spectrum. Defaults to the
        longest wavelength in the raw file.
    model_unit_wavelength : '~astropy.units.Unit' [length], default=u.AA
        Wavelength unit of the model.
    model_unit_flux : astropy.units.Unit [flux], default=u.Unit('erg cm-2 s-1 cm-1')
        Flux unit of the model.
    target_unit_wavelength : astropy.units.Unit [length], default=u.um
        Wavelength unit of the binned spectrum.
    target_unit_flux : astropy.units.Unit [flux], default=u.Unit('W m-2 um-1')
        Flux unit of the binned spectrum.

    Returns
    -------
    binned_wavelength : astropy.units.quantity.Quantity [length]
        Wavelength points of the binned spectrum.
    binned_flux : astropy.units.quantity.Quantity [flux]
        Flux points of the new spectrum.
    """
    fh5 = h5py.File(path, 'r')
    wl = fh5['PHOENIX_SPECTRUM/wl'][()] * model_unit_wavelength
    fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()] * model_unit_flux
    wl = wl.to(target_unit_wavelength)
    fl = fl.to(target_unit_flux)
    if lam1 is None:
        lam1 = np.min(wl)
    if lam2 is None:
        lam2 = np.max(wl)
    binned_wavelengths = get_wavelengths(resolving_power,
                                         to_float(
                                             lam1, target_unit_wavelength),
                                         to_float(lam2, target_unit_wavelength)) * target_unit_wavelength
    region_to_bin = (wl >= lam1) & (wl <= lam2)
    wl = wl[region_to_bin]
    fl = fl[region_to_bin]
    binned_flux = bin_spectra(
        np.array(wl.to_value(target_unit_wavelength)),
        np.array(fl.to_value(target_unit_flux)),
        np.array(binned_wavelengths.to_value(target_unit_wavelength))
    )
    return binned_wavelengths[:-1], binned_flux*target_unit_flux

def bin_spectra(wl_old:np.array,fl_old:np.array,wl_new:np.array):
    """
    Bin spectra

    This is a generic binning funciton.

    Parameters
    ----------
    wl_old : np.ndarray
        The original wavelength values.
    fl_old : np.ndarray
        The original flux values.
    wl_new : np.ndarray
        The new wavelength values.
    
    Returns
    -------
    fl_new : np.ndarray
        The new flux values.
    """
    binned_flux = []
    for i in range(len(wl_new) - 1):
        lam_cen = wl_new[i]
        upper = 0.5*(lam_cen + wl_new[i+1])
        if i == 0:
            # dl = upper - lam_cen # uncomment to sample blue of first pixel
            lower = lam_cen  # - dl
        else:
            lower = 0.5*(lam_cen + wl_new[i-1])
        reg = (wl_old >= lower) & (wl_old < upper)
        binned_flux.append(fl_old[reg].mean())
    binned_flux = np.array(binned_flux)
    return binned_flux




def get_phoenix_path(teff: Union[float, int]) -> Path:
    """
    Get PHOENIX path

    Get the path the PHOENIX model corresponding to a desired temperature.

    Parameters
    ----------
    teff : float or int
        Effective temperature in Kelvin.

    Returns
    -------
    pathlib.Path
        Path of PHOENIX spectrum file.

    """
    filename = f'lte0{teff:.0f}-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'
    return RAW_PHOENIX_PATH / filename


def get_binned_filename(teff: Union[float, int]) -> str:
    """
    Get binned filename

    Get the filename of a binned spectrum.

    Parameters
    ----------
    teff : float or int
        The effective temperature of the spectrum in Kelvin

    Returns
    -------
    str
        The filename of the binned spectrum.

    """
    return f'binned{teff:.0f}StellarModel.txt'


def write_binned_spectrum(wavelength: u.Quantity, flux: u.Quantity, filename: str,
                          path: Path = Path('./binned_data/')) -> None:
    """
    Write binned spectrum

    Write out the binned spectrum to file.

    Parameters
    ----------
    wavelength : astropy.units.quantity.Quantity [wavelength]
        An array of binned wavelength coordinates.
    flux : astropy.units.quantity.Quantity [flambda]
        An array of binned flux values.
    filename : str
        The name of the file to write.
    path : pathlib.Path
        The path to binned data.
    """
    if not path.exists():
        path.mkdir()
    with open(path/filename, 'w', encoding='UTF-8') as file:
        wavelength_unit_str = str(wavelength.unit)
        flux_unit_str = str(flux.unit)
        file.write(f'wavelength[{wavelength_unit_str}], flux[{flux_unit_str}]')
        for wl, fl in zip(wavelength.value, flux.value):
            file.write(f'\n{wl:.6e}, {fl:.6e}')


def get_cached_file_dir(R:int):
    """
    Get the filename of a cached spectrum

    Parameters
    ----------
    R : int
        The resolving power of the spectrum.
    
    Returns
    -------
    pathlib.Path
        The path to the spectrum.
    """
    return Path(BINNED_PHOENIX_PATH) / f'R_{R:0>6}'

def read_binned_spectrum(filename: str,
                         path: Path = Path('./binned_data/')
                         ) -> Tuple[u.Quantity, u.Quantity]:
    """
    Read binned spectrum

    Read the binned spectrum from a file

    Parameters
    ----------
    filename : str
        The name of the file to read
    path : pathlib.Path
        The path to the directory containing `filename`

    Returns
    -------
    binned_wavelength : astropy.units.quantity.Quantity [length]
        Wavelength points of the binned spectrum.
    binned_flux : astropy.units.quantity.Quantity [flux]
        Flux points of the binned spectrum.
    """
    full_path = path / filename
    data = pd.read_csv(full_path)
    wave_col = data.columns[0]
    flux_col = data.columns[1]
    wave_unit_str = wave_col.split('[')[1][:-1]
    flux_unit_str = flux_col.split('[')[1][:-1]
    wavelength = data[wave_col].values * u.Unit(wave_unit_str)
    flux = data[flux_col].values * u.Unit(flux_unit_str)
    return wavelength, flux

def bin_cached_model(teff: Union[float, int], file_name_writer: Callable = get_binned_filename,
                      binned_path: Path = Path('./binned_data/'),
                      resolving_power: int = 50, lam1: u.Quantity = None, lam2: u.Quantity = None,
                      model_unit_wavelength: u.Unit = u.AA,
                      model_unit_flux: u.Unit = u.Unit('erg cm-2 s-1 cm-1'),
                      target_unit_wavelength: u.Unit = u.um,
                      target_unit_flux: u.Unit = u.Unit('W m-2 um-1')) -> None:
    """
    Bin cached pre-binned model.

    Parameters
    ----------
    teff : float or int
        Effective temperature in Kelvin.
    file_name_writer : callable, default=VSPEC.stellar_spectra.get_binned_filename
        A function that maps teff to filename.
    binned_path : pathlib.Path
        Path to binned data.
    resolving_power : int, default=50
        Resolving power of the binned spectrum.
    lam1 : astropy.units.quantity.Quantity [length], default=None
        Starting wavelength of binned spectrum.
    lam2 : astropy.units.quantity.Quantity [length], default=None
        Ending wavelength of binned spectrum.
    model_unit_wavelength : '~astropy.units.Unit' [length], default=u.AA
        Wavelength unit of the model.
    model_unit_flux : astropy.units.Unit [flux], default=u.Unit('erg cm-2 s-1 cm-1')
        Flux unit of the model.
    target_unit_wavelength : astropy.units.Unit [length], default=u.um
        Wavelength unit of the binned spectrum.
    target_unit_flux : astropy.units.Unit [flux], default=u.Unit('W m-2 um-1')
        Flux unit of the binned spectrum.
    
    Notes
    -----
    This function has the same signature as `bin_phoenix_model`.
    """

    wavelength,flux = bin_from_cache(teff,resolving_power,lam1,lam2,
                            model_unit_wavelength,model_unit_flux,
                            target_unit_wavelength,target_unit_flux)
    write_binned_spectrum(
        wavelength, flux, file_name_writer(teff), path=binned_path)
    



def bin_phoenix_model(teff: Union[float, int], file_name_writer: Callable = get_binned_filename,
                      binned_path: Path = Path('./binned_data/'),
                      resolving_power: int = 50, lam1: u.Quantity = None, lam2: u.Quantity = None,
                      model_unit_wavelength: u.Unit = u.AA,
                      model_unit_flux: u.Unit = u.Unit('erg cm-2 s-1 cm-1'),
                      target_unit_wavelength: u.Unit = u.um,
                      target_unit_flux: u.Unit = u.Unit('W m-2 um-1')) -> None:
    """
    Bin PHOENIX model

    Bin a raw PHOENIX model and write it to file.

    Parameters
    ----------
    teff : float or int
        Effective temperature in Kelvin.
    file_name_writer : callable, default=VSPEC.stellar_spectra.get_binned_filename
        A function that maps teff to filename.
    binned_path : pathlib.Path
        Path to binned data.
    resolving_power : int, default=50
        Resolving power of the binned spectrum.
    lam1 : astropy.units.quantity.Quantity [length], default=None
        Starting wavelength of binned spectrum.
    lam2 : astropy.units.quantity.Quantity [length], default=None
        Ending wavelength of binned spectrum.
    model_unit_wavelength : '~astropy.units.Unit' [length], default=u.AA
        Wavelength unit of the model.
    model_unit_flux : astropy.units.Unit [flux], default=u.Unit('erg cm-2 s-1 cm-1')
        Flux unit of the model.
    target_unit_wavelength : astropy.units.Unit [length], default=u.um
        Wavelength unit of the binned spectrum.
    target_unit_flux : astropy.units.Unit [flux], default=u.Unit('W m-2 um-1')
        Flux unit of the binned spectrum.
    """
    raw_path = get_phoenix_path(teff)
    wavelength, flux = bin_raw_data(raw_path, resolving_power=resolving_power, lam1=lam1, lam2=lam2,
                                    model_unit_wavelength=model_unit_wavelength,
                                    model_unit_flux=model_unit_flux,
                                    target_unit_wavelength=target_unit_wavelength,
                                    target_unit_flux=target_unit_flux)
    write_binned_spectrum(
        wavelength, flux, file_name_writer(teff), path=binned_path)


def interpolate_spectra(target_teff: u.Quantity,
                        teff1: u.Quantity, wave1: u.Quantity, flux1: u.Quantity,
                        teff2: u.Quantity, wave2: u.Quantity, flux2: u.Quantity
                        ) -> Tuple[u.Quantity, u.Quantity]:
    """
    Interpolate spectra

    Use scipy.interpolate.interp2d to generate a spectrum given any
     `target_teff` between `teff1` and `teff2`.

    Parameters
    ----------
    target_teff : astropy.units.Quantity
        Teff of final spectrum.
    teff1 : astropy.units.Quantity
        First Teff to use in interpolation.
    wave1 : astropy.units.Quantity
        First wavelengths to use in interpolation.
    flux1 : astropy.units.Quantity
        First flux to use in interpolation.
    teff2 : astropy.units.Quantity
        Second Teff to use in interpolation.
    wave2 : astropy.units.Quantity
        Second wavelengths to use in interpolation.
    flux2 : astropy.units.Quantity
        Second flux to use in interpolation.

    Returns
    -------
    astropy.units.Quantity
        Wavelength of final spectrum. Identical to `wave1`.
    astropy.units.Quantity
        Interpolated flux with teff `target_teff`.

    Raises
    ------
    ValueError
        If `wave1` is not close to `wave2`.
    """
    if not np.all(isclose(wave1, wave2, tol=1e-6*wave1[0])):
        raise ValueError(
            'Cannot interpolate between spectra that do not share a wavelength axis.')
    flux_unit = flux1.unit
    interp = RegularGridInterpolator(
        ([to_float(teff1, u.K), to_float(teff2, u.K)],wave1),
        [to_float(flux1, flux_unit), to_float(flux2, flux_unit)]
    )
    return wave1, interp((to_float(target_teff, u.K),wave1),'linear') * flux_unit


def blackbody(wavelength: u.Quantity, teff: u.Quantity, area: u.Quantity, distance: u.Quantity,
              target_unit_flux: u.Unit = u.Unit('W m-2 um-1')) -> u.Quantity:
    """
    Blackbody

    Generate a blackbody spectrum.

    Parameters
    ----------
    wavelength : astropy.units.Quantity
        Wavelengths at which to sample.
    teff : astropy.units.Quantity
        Temperature of the blackbody.
    area : astropy.units.Quantity
        Area of the body.
    distance : astropy.units.Quantity
        Distance from the observer.
    target_unit_flux : astropy.units.Unit
        Unit to cast the flux to.

    Returns
    -------
    flux : astropy.units.Quantity
        The flux of the blackbody spectrum
    """
    angular_size = (np.pi * area/distance**2 * u.steradian).to(u.arcsec**2)
    A = 2 * c.h * c.c**2/wavelength**5
    B = np.exp((c.h*c.c)/(wavelength*c.k_B*teff)) - 1
    flux = (A/B * angular_size/u.steradian).to(target_unit_flux)
    return flux
