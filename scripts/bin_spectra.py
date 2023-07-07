"""
Script to bin spectra to common resolving powers
to save the user time.

"""
from pathlib import Path
from astropy import units as u
from tqdm.auto import tqdm

from VSPEC.files import BINNED_PHOENIX_PATH
from VSPEC.stellar_spectra import bin_phoenix_model, get_binned_filename,bin_raw_data, get_phoenix_path, write_binned_spectrum,get_cached_file_dir


RESOLVING_POWERS = [1000]
TEFFS = [i*100 for i in range(23,40)]

def bin_spec(R:int,teff:int):
    wl,fl = bin_raw_data(
            get_phoenix_path(teff),
            R,
            None,
            None,
            u.AA,
            u.Unit('erg cm-2 s-1 cm -1'),
            u.AA,
            u.Unit('erg cm-2 s-1 cm -1'),
            False
        )
    write_binned_spectrum(wl,fl,get_binned_filename(teff),get_cached_file_dir(R))

if __name__ in '__main__':
    for R in RESOLVING_POWERS:
        print(f'R={R}')
        for teff in tqdm(TEFFS,desc='binning',total=len(TEFFS)):
            bin_spec(R,teff)