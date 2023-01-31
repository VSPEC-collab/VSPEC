from pathlib import Path
from astropy import units as u, constants as c
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp2d

from VSPEC.helpers import to_float

RAW_PHOENIX_PATH = Path(__file__).parent / '..' / 'NextGenModels' / 'RawData'

def get_wavelengths(R, lam1,lam2):
    """get wavelengths
    Get wavelength points given a resolving power and a desired spectral range.
    Provides one more point than PSG, but is otherwise identical (i.e. a PSG 
    spectrum will have wavelength points `lam[:-1]`)

    Args:
        R (int): resolving power
        lam1 (float): initial wavelength
        lam2 (float): final wavelength
    
    Returns:
        (np.array): wavelength points
    """
    lam = lam1
    lams = [lam]
    while lam < lam2:
        dlam = lam / R
        lam = lam + dlam
        lams.append(lam)
    lams = np.array(lams)
    return lams

def bin_raw_data(path,R=50,lam1=None, lam2=None,
                model_unit_wavelength = u.AA,
                model_unit_flux = u.Unit('erg cm-2 s-1 cm-1'),
                target_unit_wavelength = u.um,
                target_unit_flux = u.Unit('W m-2 um-1')):
    """bin raw data
    Read in a raw spectrum and bin it to the desired resolving power

    Args:
        path (str or Path object): location of the model spectrum
        R=50 (int): resolving power of the binned spectrum
        lam1=None (astropy.units.quantity.Quantity [length]): starting wavelength of binned spectrum
        lam2=None (astropy.units.quantity.Quantity [length]): ending wavelength of binned spectrum
        model_unit_wavelength (astropy.units.Unit [length]): wavelength unit of the model
        model_unit_flux (astropy.units.Unit [flux]): flux unit of the model
        target_unit_wavelength (astropy.units.Unit [length]): wavelength unit of the binned spectrum
        target_unit_flux (astropy.units.Unit [flux]): flux unit of the binned spectrum

    Returns:
        (astropy.units.quantity.Quantity [length]): wavelength points of the binned spectrum
        (astropy.units.quantity.Quantity [flux]): flux points of the new spectrum

    """
    fh5 = h5py.File(path,'r')
    wl = fh5['PHOENIX_SPECTRUM/wl'][()] * model_unit_wavelength
    fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()] * model_unit_flux
    wl = wl.to(target_unit_wavelength)
    fl = fl.to(target_unit_flux)
    if lam1 is None:
        lam1 = min(wl)
    if lam2 is None:
        lam2 = max(wl)
    binned_wavelengths = get_wavelengths(R,
                    to_float(lam1,target_unit_wavelength),
                    to_float(lam2,target_unit_wavelength)) * target_unit_wavelength
    region_to_bin = (wl >= lam1) & (wl <= lam2)
    wl = wl[region_to_bin]
    fl = fl[region_to_bin]
    binned_flux = []
    for i in range(len(binned_wavelengths) - 1):
        lam_cen = binned_wavelengths[i]
        upper = 0.5*(lam_cen + binned_wavelengths[i+1])
        if i==0:
            # dl = upper - lam_cen # uncomment to sample blue of first pixel
            lower = lam_cen #- dl
        else:
            lower = 0.5*(lam_cen + binned_wavelengths[i-1])
        reg = (wl >= lower) & (wl < upper)
        binned_flux.append(to_float(fl[reg].mean(),target_unit_flux))
    binned_flux = np.array(binned_flux) * target_unit_flux
    binned_wavelengths = binned_wavelengths[:-1]
    return binned_wavelengths, binned_flux

def get_phoenix_path(teff):
    """get phoenix path
    Get the path the PHOENIX model corresponding to a desired temperature

    Args:
        teff (float or int): effective temperature in Kelvin
    
    Returns:
        (pathlib.Path): path of file
    """
    filename =  f'lte0{teff:.0f}-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'
    return RAW_PHOENIX_PATH / filename

def get_binned_filename(teff):
    """get binned filename
    Get the filename of a binned spectrum

    Args:
        teff (float or int): effective temperature in Kelvin
    
    Returns:
        (str) filename
    """
    return f'binned{teff:.0f}StellarModel.txt'


def write_binned_spectrum(wavelength, flux, filename,
                            path = Path('./binned_data/')):
    """write binned spectrum
    Write out the binned spectrum to file

    Args:
        wavelength (astropy.units.quantity.Quantity [length]): binned wavelength
        flux (astropy.units.quantity.Quantity [flux]): binned flux
        filename (str): name of file to write
        path (pathlib.Path): path to binned data
    
    Returns:
        None
    """
    if not path.exists():
        path.mkdir()
    with open(path/filename, 'w') as file:
        wavelength_unit_str = str(wavelength.unit)
        flux_unit_str = str(flux.unit)
        file.write(f'wavelength[{wavelength_unit_str}], flux[{flux_unit_str}]')
        for wl, fl in zip(wavelength.value,flux.value):
            file.write(f'\n{wl:.6e}, {fl:.6e}')
    
def read_binned_spectrum(filename, path = Path('./binned_data/')):
    """read binned spectrum
    Read the binned spectrum from a file
    
    Args:
        filename (str): name of file to write
        path (pathlib.Path): path to binned data
        
    Returns:
        (astropy.units.quantity.Quantity [length]): binned wavelength
        (astropy.units.quantity.Quantity [flux]): binned flux
    """
    full_path = path / filename
    data = pd.read_csv(full_path)
    wave_col  = data.columns[0]
    flux_col = data.columns[1]
    wave_unit_str = wave_col.split('[')[1][:-1]
    flux_unit_str = flux_col.split('[')[1][:-1]
    wavelength = data[wave_col].values * u.Unit(wave_unit_str)
    flux = data[flux_col].values * u.Unit(flux_unit_str)
    return wavelength, flux

def bin_phoenix_model(teff,file_name_writer = get_binned_filename,
                binned_path = Path('./binned_data/'),
                R=50,lam1=None, lam2=None,
                model_unit_wavelength = u.AA,
                model_unit_flux = u.Unit('erg cm-2 s-1 cm-1'),
                target_unit_wavelength = u.um,
                target_unit_flux = u.Unit('W m-2 um-1')):
    """bin phoenix model
    Bin a raw PHOENIX model and write it to file

    Args:
        teff (float or int): effective temperature in Kelvin
        file_name_writer=VSPEC.stellar_spectra.get_binned_filename (callable): method that maps teff to filename
        binned_path (pathlib.Path): path to binned data
        R=50 (int): resolving power of the binned spectrum
        lam1=None (astropy.units.quantity.Quantity [length]): starting wavelength of binned spectrum
        lam2=None (astropy.units.quantity.Quantity [length]): ending wavelength of binned spectrum
        model_unit_wavelength (astropy.units.Unit [length]): wavelength unit of the model
        model_unit_flux (astropy.units.Unit [flux]): flux unit of the model
        target_unit_wavelength (astropy.units.Unit [length]): wavelength unit of the binned spectrum
        target_unit_flux (astropy.units.Unit [flux]): flux unit of the binned spectrum

    Returns:
        None
    """
    raw_path = get_phoenix_path(teff)
    wavelength, flux = bin_raw_data(raw_path,R=R,lam1=lam1,lam2=lam2,
                                    model_unit_wavelength=model_unit_wavelength,
                                    model_unit_flux=model_unit_flux,
                                    target_unit_wavelength=target_unit_wavelength,
                                    target_unit_flux=target_unit_flux)
    write_binned_spectrum(wavelength,flux,file_name_writer(teff), path=binned_path)


def interpolate_spectra(target_teff,
                        teff1,wave1,flux1,
                        teff2,wave2,flux2):
    """
    interpolate spectra

    Use scipy.interpolate.interp2d to generate a spectrum given any `target_teff` between `teff1` and `teff2`

    Args:
        target_teff (Quantity): teff of final spectrum
        teff1 (Quantity): first teff to use in interpolation
        wave1 (Quantity): first wavelengths to use in interpolation
        flux1 (Qunatity): first flux to use in interpolation
        teff2 (Quantity): second teff to use in interpolation
        wave2 (Quantity): second wavelengths to use in interpolation
        flux2 (Qunatity): second flux to use in interpolation
    
    Returns:
        (Quantity): wave1
        (Quantity): Interpolated flux with teff `target_teff`
    """
    assert np.all(wave1==wave2)
    flux_unit = flux1.unit
    interp = interp2d(wave1,[to_float(teff1,u.K),to_float(teff2,u.K)],
                    [to_float(flux1,flux_unit),to_float(flux2,flux_unit)])
    return wave1, interp(wave1,to_float(target_teff,u.K)) * flux_unit


def blackbody(wavelength,teff,area,distance,
                target_unit_flux = u.Unit('W m-2 um-1')):
    """
    blackbody

    Generate a blackbody spectrum

    Args:
        wavelength (Quantity): wavelengths at which to sample
        teff (Quantity): Teff of the blackbody (actually just T)
        area (Quantity): Area of the body
        distance (Quantity): Distance from the observer
        target_unit_flux (Unit): Unit to cast the flux to
    """
    angular_size = (np.pi * area/distance**2 * u.steradian).to(u.arcsec**2)
    A = 2 * c.h * c.c**2/wavelength**5
    B = np.exp( (c.h*c.c)/(wavelength*c.k_B*teff) ) - 1
    flux = (A/B * angular_size/u.steradian).to(target_unit_flux)
    return flux
