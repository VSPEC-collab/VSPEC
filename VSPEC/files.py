"""VSPEC files module

This module standardizes all of the file handling
done by VSPEC.
"""
from typing import Dict
from pathlib import Path
import warnings


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


def build_directories(data_path: Path) -> Dict[str,Path]:
    """
    Build VSPEC directory structure

    Build the file system for a run if VSPEC and
    return a dictionary representing that structure.

    Parameters
    ----------
    data_path : pathlib.Path
        The path to write data for this VSPEC run.

    Returns
    -------
    dict
        Mapping to each of the `VSPEC` output subdirectories
    """
    parent_folder = data_path
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
    This function unifies all of the filename handling to a single piece of code.

    Parameters
    ----------
    N : int
        The 'phase number' of the file. Represents the iteration of the simulation.
    n_zfill : int
        The ``zfill()`` convention to use for zero-padding the phase number.
    ext : str
        The file extension, such as ``'.rad'`` for rad files.
    
    Returns
    -------
    str
        The filename
    
    Warns
    -----
    RuntimeWarning
        If `N` has more digits than `n_zfill` can accommodate.
    """
    if N > (10**(n_zfill)-1):
        warnings.warn(f'zfill of {n_zfill} not high enough for phase {N}',RuntimeWarning)
    return f'phase{str(N).zfill(n_zfill)}.{ext}'