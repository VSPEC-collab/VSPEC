"""VSPEC files module

This module standardizes all of the file handling
done by VSPEC.
"""
from pathlib import Path
import warnings

N_ZFILL = 5
"""int
    `__width` argument for filename `str.zfill()` calls. When writing 
    and reading files in the `VSPEC` output, this variable specifies
    the number of leading zeros to use in the filename.
"""

RAW_PHOENIX_PATH = Path(__file__).parent / 'data' / 'NextGenModels' / 'RawData'
"""str
    The path to the raw PHOENIX stellar models.
"""

EXAMPLE_GCM_PATH = Path(__file__).parent / 'data' / 'GCMs'
"""str
    The path to example GCMs.
"""

MOLEC_DATA_PATH = Path(__file__).parent / 'data' / 'molec.json'
"""str
    The path to the file containing molecular data for analysis.
"""


def check_and_build_dir(path: Path) -> None:
    """
    Check and build directory.

    Check if a path exists, if not, create it.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to check and create
    """
    if isinstance(path, str):
        path = Path(str)
    if path.exists():
        pass
    else:
        path.mkdir()


def build_directories(name_of_run: str, path: Path = Path('.')) -> dict:
    """
    Build VSPEC directory structure

    Build the file system for a run if VSPEC.
    By default it is run in `./<name_of_run>`
    as specified by `ParamModel.star_name`

    Parameters
    ----------
    name_of_run : str
        Name of top-level VSPEC model output directory
    path : pathlib.Path, default=Path('.')
        Path at which to create directory structure,
        defaults to `.`

    Returns
    -------
    dict
        Mapping to each of the `VSPEC` output subdirectories
    """
    parent_folder = path / name_of_run
    check_and_build_dir(parent_folder)
    data_folder = parent_folder / 'Data'
    check_and_build_dir(data_folder)
    binned_spectra_folder = data_folder / 'binned_data'
    check_and_build_dir(binned_spectra_folder)
    all_spectra_values_folder = data_folder / 'AllModelSpectraValues'
    check_and_build_dir(all_spectra_values_folder)
    psg_combined_spectra_folder = data_folder / 'PSGCombinedSpectra'
    check_and_build_dir(psg_combined_spectra_folder)
    psg_thermal_spectra_folder = data_folder / 'PSGThermalSpectra'
    check_and_build_dir(psg_thermal_spectra_folder)
    psg_noise_folder = data_folder / 'PSGNoise'
    check_and_build_dir(psg_noise_folder)
    psg_layers_folder = data_folder / 'PSGLayers'
    check_and_build_dir(psg_layers_folder)
    psg_configs_folder = data_folder / 'PSGConfig'
    check_and_build_dir(psg_configs_folder)

    directories_dict = {'parent': parent_folder,
                        'data': data_folder,
                        'binned': binned_spectra_folder,
                        'all_model': all_spectra_values_folder,
                        'psg_combined': psg_combined_spectra_folder,
                        'psg_thermal': psg_thermal_spectra_folder,
                        'psg_noise': psg_noise_folder,
                        'psg_layers': psg_layers_folder,
                        'psg_configs': psg_configs_folder}
    return directories_dict


def get_filename(N:int, n_zfill:int, ext:str):
    """
    Get the filename that a PSG output is written to.
    This does not handle the full path, but just unifies
    all of the filename handling to a single piece of code.

    Parameters
    ----------
    N : int
        The 'phase number' of the file. Basically, which iteration
        of the simulation is this?
    n_zfill : int
        The `zfill()` convention to use.
    ext : str
        The file extension. E.g. `'.rad'` for rad files.
    
    Returns
    -------
    str
        The filename
    
    Warns
    -----
    RuntimeWarning
        If `N` has more than `n_zfill` digits.
    """
    if N > (10**(n_zfill)-1):
        warnings.warn(f'zfill of {n_zfill} not high enough for phase {N}',RuntimeWarning)
    return f'phase{str(N).zfill(n_zfill)}.{ext}'